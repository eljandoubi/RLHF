class SummableDict(dict):
    def __add__(self, other: dict) -> "SummableDict":
        if not isinstance(other, dict):
            return NotImplemented
        result = SummableDict(self)
        for key, value in other.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value
        return result
    
    def __truediv__(self, divisor: float) -> "SummableDict":
        if divisor == 0:
            raise ValueError("Divisor cannot be zero.")
        result = SummableDict()
        for key, value in self.items():
            if hasattr(value, "__truediv__"):
                result[key] = value / divisor
        return result
    
    

def dict_mean(list_of_dicts: list[dict]) -> SummableDict:
    if not list_of_dicts:
        return SummableDict()
    total = sum(list_of_dicts, SummableDict())
    n = len(list_of_dicts)
    mean = total / n
    mean["num_samples"] = n
    return mean

