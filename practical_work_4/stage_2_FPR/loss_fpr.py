# -*- coding: utf-8 -*-
"""
Loss functions for False Positive Reduction Network with class imbalance handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] - raw logits
            targets: [B] - class labels
        """
        # Convert to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of correct class
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply focal weight
        loss = focal_weight * ce_loss
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_weight = targets.float() * self.alpha + (1 - targets.float()) * (1 - self.alpha)
            loss = alpha_weight * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss
    """
    def __init__(self, pos_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 2] - logits for 2 classes
            targets: [B] - class labels (0 or 1)
        """
        # Convert to binary classification
        probs = F.softmax(inputs, dim=1)[:, 1]  # Probability of class 1
        
        # BCE with logit
        loss = -(self.pos_weight * targets * torch.log(probs + 1e-7) + 
                (1 - targets) * torch.log(1 - probs + 1e-7))
        
        return loss.mean()


def get_loss_function(config):
    """
    Get loss function based on config
    """
    if config.get('use_focal_loss', False):
        print("Using Focal Loss")
        criterion = FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        )
    elif config.get('use_weighted_loss', True):
        print("Using Weighted Cross Entropy Loss")
        # Calculate class weights
        pos_weight = config.get('pos_weight', 10.0)
        
        # Method 1: Use weighted CrossEntropyLoss
        class_weights = torch.tensor([1.0, pos_weight])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"  Class weights: [1.0, {pos_weight}]")
    else:
        print("Using standard Cross Entropy Loss")
        criterion = nn.CrossEntropyLoss()
    
    return criterion


class MetricsCalculator:
    """Calculate classification metrics"""
    
    @staticmethod
    def calculate_metrics(predictions, labels):
        """
        Calculate accuracy, precision, recall, F1
        
        Args:
            predictions: [B, 2] - logits or probabilities
            labels: [B] - ground truth labels
        
        Returns:
            dict with metrics
        """
        # Get predicted classes
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Calculate metrics
        correct = (pred_classes == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        
        # For binary classification
        # True Positives, False Positives, True Negatives, False Negatives
        tp = ((pred_classes == 1) & (labels == 1)).sum().item()
        fp = ((pred_classes == 1) & (labels == 0)).sum().item()
        tn = ((pred_classes == 0) & (labels == 0)).sum().item()
        fn = ((pred_classes == 0) & (labels == 1)).sum().item()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        # Specificity
        specificity = tn / (tn + fp + 1e-7)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }


if __name__ == "__main__":
    # Test loss functions
    
    # Dummy data
    batch_size = 16
    num_classes = 2
    
    # Imbalanced: 2 positives, 14 negatives
    logits = torch.randn(batch_size, num_classes)
    labels = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Test Focal Loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss1 = focal_loss(logits, labels)
    print(f"Focal Loss: {loss1.item():.4f}")
    
    # Test Weighted CE Loss
    weighted_ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]))
    loss2 = weighted_ce(logits, labels)
    print(f"Weighted CE Loss: {loss2.item():.4f}")
    
    # Test Metrics
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_metrics(logits, labels)
    print(f"\nMetrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
