import numpy as np


class NASWOT:
    def __init__(self, data_loader, batch_size):
        self.data_loader = data_loader
        self.batch_size = batch_size

    def init_hook(self, model):
        model.K = np.zeros((self.batch_size, self.batch_size))

        def counting_forward_hook(module, inp, out):
            try:
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                x = (inp > 0).float()
                K = x @ x.t()
                K2 = (1. - x) @ (1. - x.t())
                model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
            except Exception as e:
                pass

        def counting_backward_hook(module, inp, out):
            module.visited_backwards = True

        for name, module in model.named_modules():
            if 'ReLU' in str(type(module)):
                # hooks[name] = module.register_forward_hook(counting_hook)
                module.register_forward_hook(counting_forward_hook)
                module.register_backward_hook(counting_backward_hook)

    def predict(self, model, iter=1):
        i = 0
        self.init_hook(model)
        for batch, (X, y) in enumerate(self.data_loader):
            model(X)
            s = self._get_score(model.K)
            i += 1

            if i >= iter:
                break

        return self._get_score(model.K)

    @staticmethod
    def _get_score(K, y=None):
        s, ld = np.linalg.slogdet(K)
        return ld