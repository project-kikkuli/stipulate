import pytest
from sqlmodel import Field, Session, SQLModel, create_engine, select

from stipulate import Explorer, action, invariant, seed


class LifecycleItem(SQLModel, table=True):
    id: int = Field(primary_key=True)
    valid: bool = True


@seed(LifecycleItem)
def initial_item():
    return LifecycleItem(id=1)


def insert_item(db: Session):
    if db.get(LifecycleItem, 2) is None:
        db.add(LifecycleItem(id=2))
        db.commit()


def delete_item(db: Session):
    item = db.get(LifecycleItem, 1)
    if item is not None:
        db.delete(item)
        db.commit()


def invalidate_inserted_item(db: Session):
    item = db.get(LifecycleItem, 2)
    if item is not None:
        item.valid = False
        db.commit()


def fail_after_delete(db: Session):
    assert db.get(LifecycleItem, 1) is not None, "follow-up observed deletion"


@invariant
def all_items_valid(db: Session):
    assert all(item.valid for item in db.exec(select(LifecycleItem)).all())


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine, tables=[LifecycleItem.__table__])
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.mark.parametrize(
    "first,second,expected_kind",
    [(insert_item, invalidate_inserted_item, "custom"),
     (delete_item, fail_after_delete, "exception")],
)
def test_explores_actions_after_row_membership_changes(db, first, second, expected_kind):
    result = Explorer(
        models=[LifecycleItem],
        seeds=[initial_item],
        actions=[action(fn=fn, params={}) for fn in (first, second)],
        invariants=[all_items_valid],
        db=db,
        budget=30,
        max_depth=2,
    ).run()

    failures = [v for v in result.violations if v.kind == expected_kind]
    assert failures, "must explore the second action after inserting/deleting a row"
    assert any(len(v.sequence) == 2 for v in failures)
    assert any(first.__name__ in v.sequence[0] and second.__name__ in v.sequence[1]
               for v in failures if len(v.sequence) == 2)
    # Branch execution and shrinking must leave the seeded state intact.
    assert [(item.id, item.valid) for item in db.exec(select(LifecycleItem)).all()] == [(1, True)]


@pytest.mark.parametrize("optimizer", ["deterministic", "hypothesis", "hybrid"])
@pytest.mark.parametrize("operation", [insert_item, delete_item])
def test_membership_changes_recheck_read_annotated_invariants(db, optimizer, operation):
    @invariant(reads=["LifecycleItem.id"])
    def exactly_one_item(db: Session):
        assert len(db.exec(select(LifecycleItem)).all()) == 1, "expected one item"

    result = Explorer(
        models=[LifecycleItem],
        seeds=[initial_item],
        actions=[action(fn=operation, params={})],
        invariants=[exactly_one_item],
        db=db,
        budget=20,
        max_depth=1,
        optimizer=optimizer,
    ).run()

    assert any(v.name == "exactly_one_item" for v in result.violations)
    assert result.invariant_coverage.get("exactly_one_item", 0) > 0
