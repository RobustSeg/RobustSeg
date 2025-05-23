"""
An implementation of the paper: "Removing Bias in Multi-modal Classifiers: Regularization by Maximizing Functional
 Entropies" NeurIPS 2020.
"""

import torch


class Perturbation:
    """
    Class that in charge of the perturbation techniques
    """
    @classmethod
    def _add_noise_to_tensor(cls, tens: torch.Tensor, over_dim: int = 0) -> torch.Tensor:
        """
        Adds noise to a tensor sampled from N(0, tens.std()).
        :param tens:
        :param over_dim: over what dim to calculate the std. 0 for features over batch,  1 for over sample.
        :return: noisy tensor in the same shape as input
        """

        return tens + torch.randn_like(tens) * tens.std(dim=over_dim)
        # return tens + torch.randn_like(tens)

    @classmethod
    def perturb_tensor(cls, tens: torch.Tensor, n_samples: int, perturbation: bool = True) -> torch.Tensor:
        """
        Flatting the tensor, expanding it, perturbing and reconstructing to the original shape.
        Note, this function assumes that the batch is the first dimension.
        :param tens:
        :param n_samples: times to perturb
        :param perturbation: False - only duplicating the tensor
        :return: tensor in the shape of [batch, samples * num_eval_samples]
        """
        tens_dim = list(tens.shape)

        tens = tens.view(tens.shape[0], -1)
        tens = tens.repeat(1, n_samples)

        tens = tens.view(tens.shape[0] * n_samples, -1)

        if perturbation:
            tens = cls._add_noise_to_tensor(tens)

        tens_dim[0] *= n_samples

        tens = tens.view(*tens_dim)
        tens.requires_grad_()

        return tens

    @classmethod
    def perturb_tensor_noise(cls, tens: torch.Tensor, noise_tens:torch.Tensor , n_samples: int, perturbation: bool = True) -> torch.Tensor:
        """
        Flatting the tensor, expanding it, perturbing and reconstructing to the original shape.
        Note, this function assumes that the batch is the first dimension.
        :param tens:
        :param n_samples: times to perturb
        :param perturbation: False - only duplicating the tensor
        :return: tensor in the shape of [batch, samples * num_eval_samples]
        """
        tens_dim = list(tens.shape)

        tens = tens.view(tens.shape[0], -1)
        tens = tens.repeat(1, n_samples)

        noise_tens = noise_tens.view(noise_tens.shape[0], -1)
        noise_tens = noise_tens.repeat(1, n_samples)

        tens = tens.view(tens.shape[0] * n_samples, -1)
        noise_tens = tens.view(noise_tens.shape[0] * n_samples, -1)

        if perturbation:
            tens = cls._add_noise_to_tensor(noise_tens)

        tens_dim[0] *= n_samples

        tens = tens.view(*tens_dim)
        tens.requires_grad_()

        return tens

    @classmethod
    def get_expanded_logits(cls, logits: torch.Tensor, n_samples: int, logits_flg: bool = True) -> torch.Tensor:
        """
        Perform Softmax and then expand the logits depends on the num_eval_samples
        :param logits_flg: whether the input is logits or softmax
        :param logits: tensor holds logits outputs from the model
        :param n_samples: times to duplicate
        :return:
        """
        if logits_flg:
            logits = torch.nn.functional.softmax(logits, dim=1)
        # expanded_logits = logits.repeat(1, n_samples)
        expanded_logits = logits.repeat(n_samples, 1, 1, 1)

        # return expanded_logits.view(expanded_logits.shape[0] * n_samples, -1)
        return expanded_logits


class Regularization(object):
    """
    Class that in charge of the regularization techniques
    """
    @classmethod
    def _get_variance(cls, loss: torch.Tensor) -> torch.Tensor:
        """
        Computes the variance along samples for the first dimension in a tensor
        :param loss: [batch, number of evaluate samples]
        :return: variance of a given batch of loss values
        """

        return torch.var(loss, dim=1)

    @classmethod
    def _get_differential_entropy(cls, loss: torch.Tensor) -> torch.Tensor:
        """
        Computes differential entropy: -E[flogf]
        :param loss:
        :return: a tensor holds the differential entropy for a batch
        """

        return -1 * torch.sum(loss * loss.log())

    @classmethod
    def _get_functional_entropy(cls, loss: torch.Tensor) -> torch.Tensor:
        """
        Computes functional entropy: E[flogf] - E[f]logE[f]
        :param loss:
        :return: a tensor holds the functional entropy for a batch
        """
        loss = torch.nn.functional.normalize(loss, p=1, dim=1)
        loss = torch.mean(loss * loss.log()) - (torch.mean(loss) * torch.mean(loss).log())

        return loss

    @classmethod
    def get_batch_statistics(cls, loss: torch.Tensor, n_samples: int, estimation: str = 'ent') -> torch.Tensor:
        """
        Calculate the expectation of the batch gradient
        :param n_samples:
        :param loss:
        :param estimation:
        :return: Influence expectation
        """
        loss = loss.reshape(-1, n_samples)

        if estimation == 'var':
            batch_statistics = cls._get_variance(loss)
            batch_statistics = torch.abs(batch_statistics)
        elif estimation == 'ent':
            batch_statistics = cls._get_functional_entropy(loss)
        elif estimation == 'dif_ent':
            batch_statistics = cls._get_differential_entropy(loss)
        else:
            raise NotImplementedError(f'{estimation} is unknown regularization, please use "var" or "ent".')

        return torch.mean(batch_statistics)

    @classmethod
    def get_batch_norm(cls, grad: torch.Tensor, loss: torch.Tensor = None, estimation: str = 'ent') -> torch.Tensor:
        """
        Calculate the expectation of the batch gradient
        :param loss:
        :param estimation:
        :param grad: tensor holds the gradient batch
        :return: approximation of the required expectation
        """
        batch_grad_norm = torch.norm(grad, p=2, dim=1)
        batch_grad_norm = torch.pow(batch_grad_norm, 2)

        if estimation == 'ent':
            batch_grad_norm = batch_grad_norm / loss

        return torch.mean(batch_grad_norm)

    @classmethod
    def _get_batch_norm(cls, grad: torch.Tensor, loss: torch.Tensor = None, estimation: str = 'ent') -> torch.Tensor:
        """
        Calculate the expectation of the batch gradient
        :param loss:
        :param estimation:
        :param grad: tensor holds the gradient batch
        :return: approximation of the required expectation
        """
        batch_grad_norm = torch.norm(grad, p=2, dim=1)
        batch_grad_norm = torch.pow(batch_grad_norm, 2)

        if estimation == 'ent':
            batch_grad_norm = batch_grad_norm / loss

        return batch_grad_norm

    @classmethod
    def _get_max_ent(cls, inf_scores: torch.Tensor, norm: float) -> torch.Tensor:
        """
        Calculate the norm of 1 divided by the information
        :param inf_scores: tensor holding batch information scores
        :param norm: which norm to use
        :return:
        """
        return torch.norm(torch.div(1, inf_scores), p=norm)

    @classmethod
    def _get_max_ent_minus(cls, inf_scores: torch.Tensor, norm: float) -> torch.Tensor:
        """
        Calculate -1 * the norm of the information
        :param inf_scores: tensor holding batch information scores
        :param norm: which norm to use
        :return:
        """
        return -1 * torch.norm(inf_scores, p=norm) + 0.1

    @classmethod
    def get_regularization_term(cls, inf_scores: torch.Tensor, norm: float = 2.0,
                                optim_method: str = 'max_ent') -> torch.Tensor:
        """
        Compute the regularization term given a batch of information scores
        :param inf_scores: tensor holding a batch of information scores
        :param norm: defines which norm to use (1 or 2)
        :param optim_method: Define optimization method (possible methods: "min_ent", "max_ent", "max_ent_minus",
         "normalized")
        :return:
        """

        if optim_method == 'max_ent':
            return cls._get_max_ent(inf_scores, norm)
        elif optim_method == 'min_ent':
            return torch.norm(inf_scores, p=norm)
        elif optim_method == 'max_ent_minus':
            return cls._get_max_ent_minus(inf_scores, norm)

        raise NotImplementedError(f'"{optim_method}" is unknown')


class RegParameters(object):
    """
    This class controls all the regularization-related properties
    """
    def __init__(self, lambda_: float = 1e-10, norm: float = 2.0, estimation: str = 'ent',
                 optim_method: str = 'max_ent', n_samples: int = 10, grad: bool = True):
        self.lambda_ = lambda_
        self.norm = norm
        self.estimation = estimation
        self.optim_method = optim_method
        self.n_samples = n_samples
        self.grad = grad