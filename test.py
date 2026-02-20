import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@torch.no_grad()
def run_test(model, params, device):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    loader  = DataLoader(test_ds, batch_size=params["batch_size"],
                         shuffle=False, num_workers=params["num_workers"])

    # Load best weights
    model.load_state_dict(torch.load(params["save_path"], map_location=device))
    model.eval()

    correct, n = 0, 0
    class_correct = [0] * params["num_classes"]
    class_total   = [0] * params["num_classes"]

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        n       += imgs.size(0)
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t]   += 1

    print(f"\n=== Test Results ===")
    print(f"Overall accuracy: {correct/n:.4f}  ({correct}/{n})\n")
    for i in range(params["num_classes"]):
        acc = class_correct[i] / class_total[i]
        print(f"  Digit {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")