from pathlib import Path

import cv2
import torch

from depth_pro import (
    Config,
    DepthPro,
)


def main() -> None:
    device = torch.device('mps')
    config = Config(checkpoint=Path(__file__).parent / 'checkpoint/depth_pro.pt')
    model = DepthPro(config, device, torch.half)

    image = cv2.cvtColor(cv2.imread('dev/data/image.jpeg'), cv2.COLOR_BGR2RGB)
    *original_size, _ = image.shape

    image = cv2.resize(image, (model.input_image_size, model.input_image_size))
    image = torch.tensor(image, device=device).permute(2, 0, 1)
    image = torch.unsqueeze(image, 0)
    image = model.input_transformation(image)

    result = model.predict(image).depth
    result /= result.max()
    result *= 255.0
    result = result.to(torch.uint8)
    result = result.cpu().numpy()
    result = cv2.resize(result, (original_size[1], original_size[0]))
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    cv2.imwrite('dev/output/depth.png', result)


if __name__ == '__main__':
    main()
