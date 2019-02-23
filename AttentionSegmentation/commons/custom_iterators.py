

class CustomIterator(object):
    def __call__(
        self,
        data: Dict[str, Iterable[Instance]],
        num_epochs: Optional[int] = None,
        shuffle: bool = True,
        for_training: bool = True,
        cuda_device: int = -1
    ):
        if self._type == "round-robin":
            generators = {}
            for lang in data:
                generators[lang] = self._iterators[lang](
                    data[lang],
                    num_epochs=None,
                    shuffle=shuffle,
                    for_training=for_training,
                    cuda_device=cuda_device
                )
            num_batches = self.get_num_batches(data)
            if num_epochs is None:
                while True:
                    yield from self._yield_round_robin_epoch(
                        generators, num_batches
                    )
            else:
                for _ in range(num_epochs):
                    yield from self._yield_round_robin_epoch(
                        generators, num_batches
                    )

    def get_num_batches(self, data):
        if self._type == "round-robin":
            return sum(
                [self._iterators[lang].get_num_batches(
                    data[lang]) for lang in data]
            )

    def _yield_round_robin_epoch(
        self,
        generators: Dict[str, DataIterator],
        num_batches: int
    ):
        for _ in range(num_batches):
            lang = self._sample_language()
            batch = next(generators[lang])
            yield batch
