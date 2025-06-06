from auffusion import AuffusionGuidance
import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    model = AuffusionGuidance(fp16=False)
    model = model.to(torch.device('cpu'))
    prompt_audio = 'a meowing cat'
    prompt_video = 'a painting of a cat, grayscale'
    spect = model.prompt_to_spec(prompt_audio, prompt_video, height=256, width=1024, num_inference_steps=10, device=torch.device('cpu'))

    print(torch.mean(spect), torch.min(spect), torch.max(spect))
    img = spect.detach().cpu().squeeze(0) 
    img = img.permute(1, 2, 0).numpy()
    # img = img.astype("16")
    img = img.clip(0, 1)
    # print(img.shape, type(img))
    # gray = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]

    plt.figure()
    plt.imshow(img)
    plt.show()

    # capito l'errore. Dobbiamo mettere fp16=False per usare doppia precisione, ma la GPU non regge, 