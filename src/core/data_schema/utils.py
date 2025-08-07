from core.constants import WRITERS, CHARS, NUM_CHARS


class IdMapper:
    _CHAR_ID_MAP = {char: id for id, char in enumerate(CHARS, 1)}
    _ID_CHAR_MAP = {v: k for k, v in _CHAR_ID_MAP.items()}
    
    @classmethod
    def chars_to_ids(cls, chars: list[str]) -> list[int]:
        return [cls._CHAR_ID_MAP[char] for char in chars]
    
    @classmethod
    def ids_to_chars(cls, ids: list[int]) -> list[str]:
        return [cls._ID_CHAR_MAP.get(id, '') for id in ids]
    
    @classmethod
    def str_to_ids(cls, s: str) -> list[int]:
        bos_id = NUM_CHARS + 1
        eos_id = NUM_CHARS + 2
        
        ids = cls.chars_to_ids(list(s))
        ids = [bos_id] + ids + [eos_id]
        return ids
    
    @classmethod
    def ids_to_str(cls, ids: list[int]) -> str:
        return ''.join(cls.ids_to_chars(ids))
    

if __name__ == "__main__":
    print(IdMapper.str_to_ids("Hello, world!"))