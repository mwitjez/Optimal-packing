from torch.nn import CrossEntropyLoss

class CustomLoss(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, output, solution):
        o = output[0]
        o = o.contiguous().view(-1, o.size()[-1])
        solution = solution.view(-1)
        return super().forward(o, solution)
