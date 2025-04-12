# %%
import pathlib
import gdown

# %%

FILES_DICT = {
  
    "mixer_lj_80.pth": {
        "path": "pretrained/mixer_lj_80.pth",
        "url": "https://drive.google.com/file/d/1YTiA6S3okiuX-_AttUhJNVgiPzVYAyjv/view?usp=sharing",
        "download": True,
    },
 
    "mixer_lj_128.pth": {
        "path": "pretrained/mixer_lj_128.pth",
        "url": "https://drive.google.com/file/d/1wVvOyaBLxqrKAssXmEYG9mszZsqEaX5R/view?usp=sharing",
        "download": True,
    },
 
    "mixer_lj_384.pth": {
        "path": "pretrained/mixer_lj_384.pth",
        "url": "https://drive.google.com/file/d/16Rq99ZmXVfiDE_nsxmUBzF3XKEOUh5wx/view?usp=sharing",
        "download": True,
    },
    
}

# %%

root_dir = pathlib.Path(__file__).parent

for file_dict in FILES_DICT.values():
    file_path = root_dir.joinpath(file_dict['path'])

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
    if file_path.exists():
        print(file_dict['path'], "already exists!")
    elif file_dict.get('download', True):
        print("Downloading ", file_dict['path'], "...")
        output_filepath = gdown.download(file_dict['url'], output=file_path.as_posix(), fuzzy=True)

