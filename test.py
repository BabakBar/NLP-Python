def find_left_boundary(a, key):
        lo, hi = 0, len(a) - 1

        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if key <= a[mid]:
                hi = mid - 1
            else:
                lo = mid + 1

        return lo if 0 <= lo < len(a) and a[lo] == key else -1
        
def find_right_boundary(a, key):
        lo, hi = 0, len(a) - 1

        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if key < a[mid]:
                hi = mid - 1
            else:
                lo = mid + 1

        return hi if 0 <= hi < len(a) and a[hi] == key else -1

def searchRange(a, key):
        return [find_left_boundary(a, key),find_right_boundary(a, key)]

 #why i need this in vscode   
if __name__ == "__main__":
    
    key = [5,7,7,8,8,10]
    result = searchRange(key, 8)
    print(result)
    