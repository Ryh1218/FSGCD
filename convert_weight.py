import torch


def convert_vitdino_weight(path):
    old_ckpts = torch.load(path, map_location="cpu")
    new_ckpts = {}

    for k, v in old_ckpts.items():
        if k.startswith("blocks."):
            spk = k.split(".")
            if ".".join(spk[2:]) == "mlp.fc1.bias":
                new_ckpts[".".join(spk[:2] + ["fc1", "bias"])] = v
            elif ".".join(spk[2:]) == "mlp.fc1.weight":
                new_ckpts[".".join(spk[:2] + ["fc1", "weight"])] = v
            elif ".".join(spk[2:]) == "mlp.fc2.bias":
                new_ckpts[".".join(spk[:2] + ["fc2", "bias"])] = v
            elif ".".join(spk[2:]) == "mlp.fc2.weight":
                new_ckpts[".".join(spk[:2] + ["fc2", "weight"])] = v
            else:
                new_ckpts[k] = v
        else:
            new_ckpts[k] = v

    assert path.endswith(".pth"), path
    new_path = path[:-4] + "_fsgcd.pth"
    torch.save(new_ckpts, new_path)
    print("Finished :", path)


if __name__ == "__main__":
    # Change the path to the pretrained weights accordingly, it will generate xx_fsgcd.pth
    # For example, path = "/root/dino_vitbase16_pretrain.pth"
    path = "{YOUR_PTH_PATH}"
    convert_vitdino_weight(path)
