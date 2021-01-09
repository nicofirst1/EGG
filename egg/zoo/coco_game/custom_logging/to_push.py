import random

from egg.core import Interaction, LoggingStrategy


class RandomLogging(LoggingStrategy):
    """
    Log strategy based on random probability
    """

    def __init__(self, logging_step: int, store_prob: float = 1, *args):
        self.store_prob = store_prob
        self.logging_step = logging_step
        self.cur_batch = 0
        super().__init__(*args)

    def filtered_interaction(
        self,
        sender_input,
        receiver_input,
        labels,
        message,
        receiver_output,
        message_length,
        aux,
    ):
        rnd = random.random()
        should_store = rnd < self.store_prob
        if self.cur_batch % self.logging_step == 0:
            should_store = True

        self.cur_batch += 1

        return Interaction(
            sender_input=sender_input if should_store else None,
            receiver_input=receiver_input if should_store else None,
            labels=labels if should_store else None,
            message=message if should_store else None,
            receiver_output=receiver_output if should_store else None,
            message_length=message_length if should_store else None,
            aux=aux,
        )
