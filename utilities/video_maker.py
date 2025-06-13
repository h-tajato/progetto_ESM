from moviefy import *
import os 

estensioni_audio = ('mp3', 'wav')
estensioni_video = ('png')

if __name__ == '__main__':
    path = 'C:\\Users\\Flexo Rodriguez\\Desktop\\my_selection'
    try:
        im_name = None
        audio_name = None
        for dir in os.listdir(path):
            local_path = path + '\\' + dir
            output_path = local_path + '.mp4'
            if os.path.isdir(local_path):
                for file in os.listdir(local_path):
                    complete_path = local_path + '\\' + file 
                    if os.path.isfile(complete_path):
                        if complete_path.split('.')[1] in estensioni_audio:
                            audio_name = complete_path
                        if complete_path.split('.')[1] in estensioni_video:
                            im_name = complete_path
                    print(f'[MAIN]\tchiamo moviefie su ({im_name}, {audio_name}, {output_path}) nella dir {local_path}')    
                    if audio_name is None and im_name is None:
                        raise RuntimeError('non ho trovato i file richiesti')
            moviefy(im_name, audio_name, output_path)
    except RuntimeError as e:
        print(e)