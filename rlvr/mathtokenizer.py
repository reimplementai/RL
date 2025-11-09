# our own tokenizer for the rlvr addition task.
# SentencePiece + BPE is less robust to noise for smaller data and models

class mathtokenizer:
    pad:int = 0
    bos:int = 1
    eos:int = 2
    tokenid_to_char = {}
    char_to_tokenid = {}
    def __init__(self):
        self.tokenid_to_char[self.pad] = 'p'
        self.char_to_tokenid['p'] = self.pad

        self.tokenid_to_char[self.bos] = 'b'
        self.char_to_tokenid['b'] = self.bos

        self.tokenid_to_char[self.eos] = 'e'
        self.char_to_tokenid['e'] = self.eos

        chars = '0123456789+=-*'
        i = len(self.tokenid_to_char)
        for ch in chars:
            self.tokenid_to_char[i] = ch
            self.char_to_tokenid[ch] = i
            i += 1

    def decode(self, ids, **kwargs):
        if isinstance(ids, int):
            return self.tokenid_to_char[ids]

        s = []
        for id in ids:
            s.append(self.tokenid_to_char[id])
        return "".join(s)
    
    def encode(self, s, **kwargs):
        ids = []
        for ch in s:
            ids.append(self.char_to_tokenid[ch])
        return ids
    
    def id_to_piece(self, id):
        return self.tokenid_to_char[id]
    
    def num_tokens(self):
        return len(self.tokenid_to_char)

    def pad_id(self):
        return self.pad
    def bos_id(self):
        return self.bos
    def eos_id(self):
        return self.eos

    def load(self, s):
        return
