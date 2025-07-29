from core.constants import WRITERS, CHARS


class IdMapper:
    _WRITER_ID_MAP = {writer: id for id, writer in enumerate(WRITERS)}
    _ID_WRITER_MAP = {v: k for k, v in _WRITER_ID_MAP.items()}
    
    _CHAR_ID_MAP = {char: id for id, char in enumerate(CHARS, 1)}
    _ID_CHAR_MAP = {v: k for k, v in _CHAR_ID_MAP.items()}
    
    @classmethod
    def writer_to_id(cls, writer: str) -> int:
        return cls._WRITER_ID_MAP[writer]
    
    @classmethod
    def chars_to_ids(cls, chars: list[str]) -> list[int]:
        return [cls._CHAR_ID_MAP[char] for char in chars]
    
    @classmethod
    def id_to_writer(cls, id: int) -> str:
        return cls._ID_WRITER_MAP[id]
    
    @classmethod
    def ids_to_chars(cls, ids: list[int]) -> list[str]:
        return [cls._ID_CHAR_MAP[id] for id in ids]
    
    @classmethod
    def str_to_ids(cls, s: str, add_bos: bool=False, add_eos: bool=False) -> list[int]:
        max_id = max(cls._CHAR_ID_MAP.values())
        bos_id = max_id + 1
        eos_id = max_id + 2
        
        ids = cls.chars_to_ids(list(s))
        if add_bos:
            ids = [bos_id] + ids
        if add_eos:
            ids = ids + [eos_id]
        return ids
    
    @classmethod
    def ids_to_str(cls, ids: list[int]) -> str:
        return ''.join(cls.ids_to_chars(ids))