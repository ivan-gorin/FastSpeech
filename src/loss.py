from torch import nn


class FastSpeechLoss(nn.Module):

    def __init__(self):
        super(FastSpeechLoss, self).__init__()
        self.duration_mse = nn.MSELoss()
        self.spec_mse = nn.MSELoss()

    def forward(self, durations, pred_durations, specs, pred_specs):
        min_len_duration = min(durations.shape[-1], pred_durations.shape[-1])
        min_len_spec = min(specs.shape[-1], pred_specs.shape[-1])
        return (self.duration_mse(pred_durations[:, :min_len_duration].exp(), durations[:, :min_len_duration]),
                self.spec_mse(pred_specs[:, :, :min_len_spec], specs[:, :, :min_len_spec]))
