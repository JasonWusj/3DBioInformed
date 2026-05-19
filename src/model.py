import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_channels = in_channels
        for channels in features:
            self.encoders.append(DoubleConv3D(prev_channels, channels))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_channels = channels

        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)
        self.bottleneck_channels = features[-1] * 2

        for channels in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(channels * 2, channels, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv3D(channels * 2, channels))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x, return_features=False):
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        bottleneck_features = x

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        logits = self.final_conv(x)
        if return_features:
            return logits, bottleneck_features
        return logits


class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = (6.0 / in_features) ** 0.5 / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class DensityEstimator3D(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_layers=3, feature_size=(16, 16, 16)):
        super().__init__()
        self.feature_size = tuple(int(v) for v in feature_size)
        self.adapt_pool = nn.AdaptiveAvgPool3d(self.feature_size)

        layers = [SirenLayer(in_channels + 1, hidden_dim, is_first=True)]
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        self.siren = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, features, t=1.0):
        batch_size = features.shape[0]
        depth, height, width = self.feature_size

        x = self.adapt_pool(features)
        x = x.flatten(2).permute(0, 2, 1)

        t_tensor = torch.full(
            (batch_size, depth * height * width, 1),
            float(t),
            device=x.device,
            dtype=x.dtype,
            requires_grad=True,
        )
        y = torch.cat([x, t_tensor], dim=-1)
        y = self.siren(y)
        u_hat = torch.sigmoid(self.output_layer(y))
        u_hat = u_hat.permute(0, 2, 1).reshape(batch_size, 1, depth, height, width)
        return u_hat, t_tensor


class BiophysicsSegModel3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]

        self.unet = UNet3D(
            in_channels=data_cfg["num_channels"],
            out_channels=data_cfg["num_classes"],
            features=model_cfg["features"],
        )
        self.density_estimator = DensityEstimator3D(
            in_channels=self.unet.bottleneck_channels,
            hidden_dim=model_cfg["density_estimator"]["hidden_dim"],
            num_layers=model_cfg["density_estimator"]["num_layers"],
            feature_size=model_cfg["density_estimator"]["feature_size"],
        )

    def forward(self, x, return_density=True):
        if not return_density:
            return self.unet(x, return_features=False)

        logits, features = self.unet(x, return_features=True)
        u_hat, t_tensor = self.density_estimator(features)
        return logits, u_hat, t_tensor


class StandardSegModel3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        self.unet = UNet3D(
            in_channels=data_cfg["num_channels"],
            out_channels=data_cfg["num_classes"],
            features=model_cfg["features"],
        )

    def forward(self, x, return_density=False):
        return self.unet(x, return_features=False)
