#Binary Search
#check commit
def binary_search(list, target):
    first, last = 0, len(list) - 1
    
    while first <= last:
        mid = (first + last) // 2
        
        if list[mid] == target:
            return mid
        elif list[mid] < target:
            first = mid + 1
        else: 
            last = mid - 1 

    return None

if __name__ == "__main__":
    
    def verify(ind):
        if ind is not None:
            print("Found the number:", ind)
        else:
            print("Not F")

nums = [1,2,3,4,5]
result = binary_search(nums, 3)
verify(result)
