import torch

class AutomaticWeightedLoss(torch.nn.Module):
    '''
        parameters: num: the number of losses
        Here is the weighting method for selecting loss. 
        The first one is manual setting. 
        The second one is automatic weight adjustment based on maximum likelihood estimation. 
        The third one is automatic weight adjustment based on maximum likelihood estimation after regularization term.
    '''
    def __init__(self, num=3):  # Adjusting for three losses
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += loss
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum