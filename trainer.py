from tqdm import tqdm
from utils import *
"""
Train and evaluate wrapper for the model
"""

class TIFTrain:
    def __init__(self, args):
        self.device = args.device
        self.metrics = None

    def compute_loss(self, R_est, t_est, R_gt, t_gt):
        I = torch.eye(3).to(self.device)
        R_loss = torch.norm(R_est @ R_gt.mT - I, p=2)
        t_loss = torch.norm(t_est-t_gt, p=2)
        return R_loss + t_loss

    def compute_metrics(self, R_est, t_est, R_gt, t_gt):
        R_RMSE = torch.sqrt(torch.nn.functional.mse_loss(R_est, R_gt))
        R_MAE = torch.nn.functional.l1_loss(R_est, R_gt)
        t_RMSE = torch.sqrt(torch.nn.functional.mse_loss(t_est, t_gt))
        t_MAE = torch.nn.functional.l1_loss(t_est, t_gt)
        return R_RMSE, R_MAE, t_RMSE, t_MAE


    def train(self, model, trainloader, optim):
        model.train()

        total_loss = 0
        iter = 0
        for i, data in tqdm(enumerate(trainloader), total=trainloader.__len__()):
            target, source, gt = data
            target = target.to(self.device)
            source = source.to(self.device)
            gt = gt.to(self.device)
            R_est, t_est = model(source, target)
            R_gt = gt[:, 0:3, 0:3]
            t_gt = gt[:, 3, 0:3]
            loss = self.compute_loss(R_est, t_est, R_gt, t_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            iter += 1
        return total_loss / iter

    def eval(self, model, testloader):
        model.eval()

        total_loss = 0
        iter = 0
        metrics = torch.zeros((1, 4))
        for i, data in tqdm(enumerate(testloader), total=testloader.__len__()):
            with torch.no_grad():
                target, source, gt = data
                target = target.to(self.device)
                source = source.to(self.device)
                gt = gt.to(self.device)
                R_est, t_est = model(source, target)
                R_gt = gt[:, 0:3, 0:3]
                t_gt = gt[:, 3, 0:3]
                loss = self.compute_loss(R_est, t_est, R_gt, t_gt)

                # Calculate RMSE and MAE for metrics
                R_RMSE, R_MAE, t_RMSE, t_MAE = self.compute_metrics(R_est, t_est, R_gt, t_gt)
                metrics = torch.cat((metrics, torch.Tensor([[R_RMSE, R_MAE, t_RMSE, t_MAE]])))

                total_loss += loss.item()
                iter += 1

        # Average the metrics - remove the first index as it is zeros
        self.metrics = torch.mean(metrics[1:, :], dim=0)
        return total_loss / iter