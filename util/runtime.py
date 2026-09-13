"""Bounded smoke runs and explicit training/evaluation provenance."""
import itertools


def validate_eval_query_floor(floor, evaluation, levels, forced=None):
    if type(floor) is not int or floor < 0:
        raise ValueError('Query floor must be a nonnegative integer')
    if floor and (not evaluation or floor not in levels or forced is not None):
        raise ValueError('Query floor requires evaluation, a configured budget level, and no forced budget')


def apply_eval_query_floor(counts, floor=0, training=False):
    if not floor:
        return counts
    if training:
        raise ValueError('Query floor must never be used during training')
    return counts.clamp(min=floor)


def update_skipped_streak(streak, applied, limit=0):
    streak = 0 if applied else streak + 1
    if limit and streak >= limit:
        raise FloatingPointError(f'No optimizer update for {streak} consecutive iterations; stopping without retry')
    return streak


def optimizer_step_statistics(local_iterations, local_updates, synchronized_meter):
    """Keep rank-local progress separate from all-rank metric totals."""
    return dict(train_iterations=local_iterations, optimizer_steps=local_updates,
                amp_skipped_steps=local_iterations - local_updates,
                global_optimizer_steps=int(synchronized_meter.total),
                global_train_iterations=int(synchronized_meter.count))


class LimitedLoader:
    """Expose the actual run length to progress meters without fetching extra batches."""
    def __init__(self, loader, limit=0):
        self.loader = loader
        self.length = min(len(loader), limit) if limit else len(loader)
        if self.length <= 0:
            raise ValueError('The selected data loader is empty')

    def __len__(self):
        return self.length

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.length)

    @property
    def dataset(self):
        return self.loader.dataset


def run_metadata(args, train_steps, full_train_steps):
    return dict(smoke_test=bool(args.max_train_steps or args.max_eval_steps or args.debug),
                epoch_complete=train_steps == full_train_steps,
                train_iterations=train_steps,
                train_split=args.train_split, eval_split=args.eval_split)


def can_select_best(args):
    # Never use a test split or a truncated evaluation to select paper checkpoints.
    return (args.eval_split in ('val', 'eval_debug') and not args.debug and
            not args.max_eval_steps and not args.max_train_steps)
