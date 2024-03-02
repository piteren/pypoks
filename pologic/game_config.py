from pypaq.lipytools.files import list_dir
import shutil
from typing import List, Optional, Tuple
import yaml

from envy import PyPoksException, GAME_CONFIGS_FD


class GameConfig:

    def __init__(
            self,
            name: str,
            table_size: int,
            table_cash_start: int,
            table_cash_sb: int,
            table_cash_bb: int,
            table_moves: List[Tuple],
    ):

        config_names = GameConfig.get_names_from_folder(GAME_CONFIGS_FD)
        if name not in config_names:
            raise PyPoksException(f'given name: \'{name}\' for GameConfig is unknown!')

        self.name = name
        self.table_size = table_size
        self.table_cash_start = table_cash_start
        self.table_cash_sb = table_cash_sb
        self.table_cash_bb = table_cash_bb
        self.table_moves = table_moves

    @staticmethod
    def get_names_from_folder(folder:str) -> List[str]:
        files = list_dir(folder)['files']
        return [fn[:-8] for fn in files if fn.endswith('_gc.yaml')]

    @staticmethod
    def get_name_from_folder(folder:str) -> Optional[str]:
        config_names = GameConfig.get_names_from_folder(folder)
        if len(config_names) == 0:
            raise PyPoksException('there is no config_file in given folder!')
        if len(config_names) > 1:
            raise PyPoksException('there are many config_files in given folder!')
        return config_names[0]

    @classmethod
    def from_name(
            cls,
            name: Optional[str]=    None,
            folder: Optional[str]=  GAME_CONFIGS_FD,
            copy_to: Optional[str]= None,
    ) -> "GameConfig":
        """ loads game config file with given name from folder,
        optionally copies it to copy_to folder, if given """

        if name is None and folder is None:
            raise PyPoksException('name or folder must be given!')

        if not name:
            name = cls.get_name_from_folder(folder)

        with open(f"{folder}/{name}_gc.yaml", "r") as stream:
            config = yaml.safe_load(stream)

        if copy_to is not None:
            shutil.copyfile(f'{folder}/{name}_gc.yaml', f'{copy_to}/{name}_gc.yaml')

        return cls(name=name, **config)

