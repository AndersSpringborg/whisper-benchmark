import os

AUDIO_FILES = {
    'en-male-1': {
        'url': 'https://upload.wikimedia.org/wikipedia/commons/5/5d/%22The_Call_of_South_Africa%22%2C_read_by_Philip_Burgers.flac',
    },
    'en-male-2': {
        'url': 'https://upload.wikimedia.org/wikipedia/commons/7/75/0_nanolead_q10.ogg',
    },
    'en-male-3': {
        'url': 'https://upload.wikimedia.org/wikipedia/commons/a/a8/12_Why_There%27s_A_Cat_Curfew_in_My_House.oga',
    },
    'en-female-1': {
        'url': 'https://upload.wikimedia.org/wikipedia/commons/9/95/Alessia_Cara%27s_voice%2C_from_Border_Crossings_on_VOA_at_Jingle_Ball_2016.mp3',
    },
    'en-female-2': {
        'url': 'https://upload.wikimedia.org/wikipedia/commons/b/b4/Jabberwocky.ogg',
    },
    'en-female-3': {
        'url': 'https://upload.wikimedia.org/wikipedia/commons/6/61/Joely_Richardson_on_the_Albert_Memorial.ogg',
    },
}
CACHE_DIR = os.path.expanduser('~/.cache/whisper-benchmark')
