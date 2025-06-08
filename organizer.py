import os 
import spettrogram as sp


path = '.\\guida'



if __name__ == '__main__':
    os.system('rm -r ./input_spectrograms/*')
    for file in os.listdir(path):
        complete_path = os.path.join(path, file)
        if os.path.isfile(complete_path):
            dir_name = file.split('.')[0]

            new_path = f'.\\input_spectrograms\\{dir_name}\\{dir_name}.png'
            new_dir = f'./input_spectrograms/{dir_name}'
            os.makedirs(f".\\input_spectrograms\\{dir_name}", exist_ok=True)
            spectrogram, shape_spec = sp.spectrify(complete_path)
            if(sp.save_spectrogram(new_path, spectrogram, clip=False)):
                print(f'[MAIN]\t{file} salvato con successo con dimensione ({shape_spec[0]}, {shape_spec[1]})')
            os.system(f'cp {complete_path} {new_dir}')


        