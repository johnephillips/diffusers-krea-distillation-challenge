import random

import torch

from diffusers import UNet2DModel
from huggingface_hub import HfApi


api = HfApi()

results = {}
# fmt: off
results["google_ddpm_cifar10_32"] = torch.tensor([
    -0.7515, -1.6883, 0.2420, 0.0300, 0.6347, 1.3433, -1.1743, -3.7467,
    1.2342, -2.2485, 0.4636, 0.8076, -0.7991, 0.3969, 0.8498, 0.9189,
    -1.8887, -3.3522, 0.7639, 0.2040, 0.6271, -2.7148, -1.6316, 3.0839,
    0.3186, 0.2721, -0.9759, -1.2461, 2.6257, 1.3557
])
results["google_ddpm_ema_bedroom_256"] = torch.tensor([
    -2.3639, -2.5344, 0.0054, -0.6674, 1.5990, 1.0158, 0.3124, -2.1436,
    1.8795, -2.5429, -0.1566, -0.3973, 1.2490, 2.6447, 1.2283, -0.5208,
    -2.8154, -3.5119, 2.3838, 1.2033, 1.7201, -2.1256, -1.4576, 2.7948,
    2.4204, -0.9752, -1.2546, 0.8027, 3.2758, 3.1365
])
results["CompVis_ldm_celebahq_256"] = torch.tensor([
    -0.6531, -0.6891, -0.3172, -0.5375, -0.9140, -0.5367, -0.1175, -0.7869,
    -0.3808, -0.4513, -0.2098, -0.0083, 0.3183, 0.5140, 0.2247, -0.1304,
    -0.1302, -0.2802, -0.2084, -0.2025, -0.4967, -0.4873, -0.0861, 0.6925,
    0.0250, 0.1290, -0.1543, 0.6316, 1.0460, 1.4943
])
results["google_ncsnpp_ffhq_1024"] = torch.tensor([
    0.0911, 0.1107, 0.0182, 0.0435, -0.0805, -0.0608, 0.0381, 0.2172,
    -0.0280, 0.1327, -0.0299, -0.0255, -0.0050, -0.1170, -0.1046, 0.0309,
    0.1367, 0.1728, -0.0533, -0.0748, -0.0534, 0.1624, 0.0384, -0.1805,
    -0.0707, 0.0642, 0.0220, -0.0134, -0.1333, -0.1505
])
results["google_ncsnpp_bedroom_256"] = torch.tensor([
    0.1321, 0.1337, 0.0440, 0.0622, -0.0591, -0.0370, 0.0503, 0.2133,
    -0.0177, 0.1415, -0.0116, -0.0112, 0.0044, -0.0980, -0.0789, 0.0395,
    0.1502, 0.1785, -0.0488, -0.0514, -0.0404, 0.1539, 0.0454, -0.1559,
    -0.0665, 0.0659, 0.0383, -0.0005, -0.1266, -0.1386
])
results["google_ncsnpp_celebahq_256"] = torch.tensor([
    0.1154, 0.1218, 0.0307, 0.0526, -0.0711, -0.0541, 0.0366, 0.2078,
    -0.0267, 0.1317, -0.0226, -0.0193, -0.0014, -0.1055, -0.0902, 0.0330,
    0.1391, 0.1709, -0.0562, -0.0693, -0.0560, 0.1482, 0.0381, -0.1683,
    -0.0681, 0.0661, 0.0331, -0.0046, -0.1268, -0.1431
])
results["google_ncsnpp_church_256"] = torch.tensor([
    0.1192, 0.1240, 0.0414, 0.0606, -0.0557, -0.0412, 0.0430, 0.2042,
    -0.0200, 0.1385, -0.0115, -0.0132, 0.0017, -0.0965, -0.0802, 0.0398,
    0.1433, 0.1747, -0.0458, -0.0533, -0.0407, 0.1545, 0.0419, -0.1574,
    -0.0645, 0.0626, 0.0341, -0.0010, -0.1199, -0.1390
])
results["google_ncsnpp_ffhq_256"] = torch.tensor([
    0.1075, 0.1074, 0.0205, 0.0431, -0.0774, -0.0607, 0.0298, 0.2042,
    -0.0320, 0.1267, -0.0281, -0.0250, -0.0064, -0.1091, -0.0946, 0.0290,
    0.1328, 0.1650, -0.0580, -0.0738, -0.0586, 0.1440, 0.0337, -0.1746,
    -0.0712, 0.0605, 0.0250, -0.0099, -0.1316, -0.1473
])
results["google_ddpm_cat_256"] = torch.tensor([
    -1.4572, -2.0481, -0.0414, -0.6005, 1.4136, 0.5848, 0.4028, -2.7330,
    1.2212, -2.1228, 0.2155, 0.4039, 0.7662, 2.0535, 0.7477, -0.3243,
    -2.1758, -2.7648, 1.6947, 0.7026, 1.2338, -1.6078, -0.8682, 2.2810,
    1.8574, -0.5718, -0.5586, -0.0186, 2.3415, 2.1251])
results["google_ddpm_celebahq_256"] = torch.tensor([
    -1.3690, -1.9720, -0.4090, -0.6966, 1.4660, 0.9938, -0.1385, -2.7324,
    0.7736, -1.8917, 0.2923, 0.4293, 0.1693, 1.4112, 1.1887, -0.3181,
    -2.2160, -2.6381, 1.3170, 0.8163, 0.9240, -1.6544, -0.6099, 2.5259,
    1.6430, -0.9090, -0.9392, -0.0126, 2.4268, 2.3266
])
results["google_ddpm_ema_celebahq_256"] = torch.tensor([
    -1.3525, -1.9628, -0.3956, -0.6860, 1.4664, 1.0014, -0.1259, -2.7212,
    0.7772, -1.8811, 0.2996, 0.4388, 0.1704, 1.4029, 1.1701, -0.3027,
    -2.2053, -2.6287, 1.3350, 0.8131, 0.9274, -1.6292, -0.6098, 2.5131,
    1.6505, -0.8958, -0.9298, -0.0151, 2.4257, 2.3355
])
results["google_ddpm_church_256"] = torch.tensor([
    -2.0585, -2.7897, -0.2850, -0.8940, 1.9052, 0.5702, 0.6345, -3.8959,
    1.5932, -3.2319, 0.1974, 0.0287, 1.7566, 2.6543, 0.8387, -0.5351,
    -3.2736, -4.3375, 2.9029, 1.6390, 1.4640, -2.1701, -1.9013, 2.9341,
    3.4981, -0.6255, -1.1644, -0.1591, 3.7097, 3.2066
])
results["google_ddpm_bedroom_256"] = torch.tensor([
    -2.3139, -2.5594, -0.0197, -0.6785, 1.7001, 1.1606, 0.3075, -2.1740,
    1.8071, -2.5630, -0.0926, -0.3811, 1.2116, 2.6246, 1.2731, -0.5398,
    -2.8153, -3.6140, 2.3893, 1.3262, 1.6258, -2.1856, -1.3267, 2.8395,
    2.3779, -1.0623, -1.2468, 0.8959, 3.3367, 3.2243
])
results["google_ddpm_ema_church_256"] = torch.tensor([
    -2.0628, -2.7667, -0.2089, -0.8263, 2.0539, 0.5992, 0.6495, -3.8336,
    1.6025, -3.2817, 0.1721, -0.0633, 1.7516, 2.7039, 0.8100, -0.5908,
    -3.2113, -4.4343, 2.9257, 1.3632, 1.5562, -2.1489, -1.9894, 3.0560,
    3.3396, -0.7328, -1.0417, 0.0383, 3.7093, 3.2343
])
results["google_ddpm_ema_cat_256"] = torch.tensor([
    -1.4574, -2.0569, -0.0473, -0.6117, 1.4018, 0.5769, 0.4129, -2.7344,
    1.2241, -2.1397, 0.2000, 0.3937, 0.7616, 2.0453, 0.7324, -0.3391,
    -2.1746, -2.7744, 1.6963, 0.6921, 1.2187, -1.6172, -0.8877, 2.2439,
    1.8471, -0.5839, -0.5605, -0.0464, 2.3250, 2.1219
])
# fmt: on

models = api.list_models(filter="diffusers")
for mod in models:
    if "google" in mod.author or mod.modelId == "CompVis/ldm-celebahq-256":
        local_checkpoint = "/home/patrick/google_checkpoints/" + mod.modelId.split("/")[-1]

        print(f"Started running {mod.modelId}!!!")

        if mod.modelId.startswith("CompVis"):
            model = UNet2DModel.from_pretrained(local_checkpoint, subfolder="unet")
        else:
            model = UNet2DModel.from_pretrained(local_checkpoint)

        torch.manual_seed(0)
        random.seed(0)

        noise = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
        time_step = torch.tensor([10] * noise.shape[0])
        with torch.no_grad():
            logits = model(noise, time_step).sample

        assert torch.allclose(
            logits[0, 0, 0, :30], results["_".join("_".join(mod.modelId.split("/")).split("-"))], atol=1e-3
        )
        print(f"{mod.modelId} has passed succesfully!!!")
