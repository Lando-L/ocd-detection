import attr


@attr.s(eq=False, frozen=True, slots=True)
class State(object):
    """
    Structure for state on the server.

    Fields:
        - `model`: A ModelWeights structure, containing Tensors or Variables.
        - `optimizer_state`: Variables of optimizer.
        - 'round_num': Current round index
    """

    model = attr.ib()
    round_num = attr.ib()
