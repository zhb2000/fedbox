import random
from typing import Any, Iterable

import torch
import torch.nn
import torch.optim
from sklearn.metrics import accuracy_score


class Evaluate:
    """
    Provide:

    - `eval_metric`: multi-class classification accuracy as default metric.
    - `evaluate`
    - `validate`
    - `test`

    Require:

    - `valid_loader` (`Iterable`)
    - `test_loader` (`Iterable`)
    - `device` (`torch.device`)
    - `model` (`torch.nn.Module`)
    """

    def eval_metric(self, output: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
        """Multi-class classification accuracy is the default `eval_metric`."""
        pred = output.argmax(dim=1)
        acc = accuracy_score(targets.numpy(), pred.numpy())
        return {'pred': pred, 'acc': acc}

    @torch.no_grad()
    def evaluate(self: Any, loader: Iterable) -> dict[str, Any]:
        self.model.to(self.device)
        self.model.eval()
        output_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            output_list.append(out.cpu())
            targets_list.append(y.cpu())
        output = torch.concat(output_list)
        targets = torch.concat(targets_list)
        result = self.eval_metric(output, targets)
        self.model.cpu()
        return {
            'output': output,
            'targets': targets,
            **result  # pred, acc
        }

    def validate(self: Any) -> dict[str, Any]:
        return self.evaluate(self.valid_loader)

    def test(self: Any) -> dict[str, Any]:
        return self.evaluate(self.test_loader)


class PersonalizedEvaluate:
    """
    Provide:

    - `eval_metric`: multi-class classification accuracy as default metric.
    - `personalized_validate`
    - `personalized_test`

    Require:

    - `device` (`torch.device`)
    - `clients` (`list`)
    """

    eval_metric = Evaluate.eval_metric

    @torch.no_grad()
    def __personalized_evaluate(self: Any, client_evaluator) -> dict[str, Any]:
        output_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []
        for client in self.clients:
            client_result: dict[str, Any] = client_evaluator(client)
            output_list.append(client_result['output'])
            targets_list.append(client_result['targets'])
        output = torch.concat(output_list)
        targets = torch.concat(targets_list)
        result = self.eval_metric(output, targets)
        return {
            'output': output,
            'targets': targets,
            **result
        }

    def personalized_validate(self) -> dict[str, Any]:
        return self.__personalized_evaluate(lambda client: client.validate())

    def personalized_test(self) -> dict[str, Any]:
        return self.__personalized_evaluate(lambda client: client.test())


class Server:
    """
    Provide:

    - `sample_clients`
    - `make_checkpoint`
    - `load_checkpoint`

    Require:

    - `model` (`torch.nn.Module`)
    - `clients` (`list`)
    - `client_join_num` (`int | None`)
    """

    def sample_clients(self: Any) -> list:
        if self.client_join_num is not None:
            return random.sample(self.clients, self.client_join_num)
        else:
            return list(self.clients)

    def make_checkpoint(self: Any, include_clients: bool = True) -> dict[str, Any]:
        checkpoint = {'current_round': self.current_round}
        if hasattr(self, 'model'):
            checkpoint['model'] = self.model.state_dict()
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self: Any, checkpoint: dict[str, Any]):
        if 'current_round' in checkpoint:
            self.current_round = checkpoint['current_round']
        if 'model' in checkpoint and hasattr(self, 'model'):
            self.model.load_state_dict(checkpoint['model'])
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)


class Client:
    """
    Provide:

    - `make_checkpoint`
    - `load_checkpoint`

    Requires:

    - `model` (`torch.nn.Module`)
    - `optimizer` (`torch.optim.Optimizer`)
    - `scheduler`
    """

    def make_checkpoint(self: Any) -> dict[str, Any]:
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

    def load_checkpoint(self: Any, checkpoint: dict[str, Any]):
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
