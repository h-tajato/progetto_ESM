import os 
import spettrogram as sp


path = '.\\guida'



if __name__ == '__main__':

    for file in os.listdir(path):
        complete_path = os.path.join(path, file)
        if os.path.isfile(complete_path):
            dir_name = file.split('.')[0]
            new_path = f'\\input_spectrograms\\{dir_name}\\{dir_name}.png'
            os.makedirs(f".\\input_spectrograms\\{dir_name}", exist_ok=True)
            spectrogram = sp.spectrify(complete_path)
            


        