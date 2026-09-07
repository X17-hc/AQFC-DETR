"""Bounded smoke runs and explicit training/evaluation provenance."""
import itertools


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
