import torch

class EnsembleWrapper():
    def __init__(self, model):
        self.model = model

    def validation_step(self, data):
        input = data['L']
        outputs = []
        for i in range(8):
            aug_input = input.clone()
            if i >= 4:
                aug_input = torch.flip(aug_input, [2])
            if i % 4 > 1:
                aug_input = torch.flip(aug_input, [3])
            if (i % 4) % 2 == 1:
                aug_input = torch.rot90(aug_input, 1, [2, 3])

            aug_output = self.model.validation_step({'L': aug_input})

            if (i % 4) % 2 == 1:
                aug_output = torch.rot90(aug_output, 3, [2, 3])
            if i % 4 > 1:
                aug_output = torch.flip(aug_output, [3])
            if i >= 4:
                aug_output = torch.flip(aug_output, [2])
            outputs.append(aug_output)
        output = torch.stack(outputs, dim=0)
        output = torch.mean(output, dim=0, keepdim=False)
        return output