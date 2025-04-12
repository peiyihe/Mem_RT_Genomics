
import numpy as np
from tqdm import tqdm

def count_below_threshold_numpy_single(random_matrix, test_events, reference_array_single, col, Threshold, sub_array_row, n_blocks):
    HD = []
    for i in range(len(test_events) - col + 1):
        read = test_events[i:i+col]    
        read =np.dot(read, random_matrix)
        read = (read > 0).astype(int)
        search = read - reference_array_single
        abs_gt_tolerance_count = np.sum(np.abs(search) > 0, axis=1)        
        HD.append(abs_gt_tolerance_count.min())
    
    count = len([x for x in HD if x < Threshold])
    return count

def count_below_threshold_numpy_v2(random_matrix, test_events, reference_array, col, Threshold, sub_array_row, n_blocks):
    votes=[]
    # for i in tqdm(range(len(reference_array)), desc="Processing", unit="iteration"):
    for i in range(len(reference_array)):
        count_result = count_below_threshold_numpy_single(random_matrix, test_events, reference_array[i], col, Threshold, sub_array_row, n_blocks)
        votes.append(count_result)
    votes_np = np.array(votes, dtype=np.int8)
    return votes_np

def has_difference_of_one(a, b):
    for number in b:
        if abs(number - a) == 1:
            return True
    return False

def find_topk_locations_numpy(arr, k=5):
    if k <= 0:
        return [], []
    
    indices = np.argpartition(-arr, k)[:k]
    values = arr[indices]
    sorted_idx = np.argsort(-values)
    
    sorted_values = values[sorted_idx]
    sorted_indices = indices[sorted_idx]
    
    return sorted_values.tolist(), sorted_indices.tolist()

def location_decide(sample_number, total_min_values_result, event_length):
    max_location = np.argmax(total_min_values_result, axis=0).item()
    topk_values, topk_indices = find_topk_locations_numpy(total_min_values_result, k=10)

    # for debug
    # print('max_vote:',total_min_values_result[max_location])
    # print('max_vote_location:',max_location)
    # print('event_length:',event_length)
    # print(topk_values)
    # print(topk_indices)

    if sample_number >= 4000:
        sum_min_1 = 8
        sum_min_2 = 5
    else:
        sum_min_1 = 8/4000*sample_number
        sum_min_2 = 5/4000*sample_number
        # sum_min_1 = min(sum_min_1,4)
        # sum_min_2 = min(sum_min_2,3)

    if(total_min_values_result[max_location]>65):   # before 200 
        final_location= max_location
        votes = total_min_values_result[max_location].cpu().item()
    else:  
        sorted_indices = sorted(topk_indices[:3])  
        sorted_indices_2 = sorted(topk_indices[:2])

        if topk_values[1]==0 and topk_values[0]>sum_min_1/2:
            final_location=max_location
            votes = topk_values[0]
        elif topk_values[1]==0 and topk_values[0]==0:
            final_location= 'N'
            votes = 'N'

        # elif topk_values[1]>0 and (topk_values[0]/topk_values[1]>2) and topk_values[0]>5: #old version
        elif topk_values[1]>0 and (topk_values[0]/topk_values[1]>=2) and topk_values[0]>sum_min_2: #read mapping
            final_location=max_location
            votes = topk_values[0]

        elif(sorted_indices[0]+1==sorted_indices[1]) and (sorted_indices[1]+1==sorted_indices[2]):
            sum=topk_values[0]+topk_values[1]+topk_values[2]
            if  sum>sum_min_1 and topk_values[3]>0 and (sum/topk_values[3]>2):
                final_location=max_location
                votes = sum
            else:
                final_location='N'
                votes = 'N'
        elif(sorted_indices_2[0]+1==sorted_indices_2[1]):
            sum=topk_values[0]+topk_values[1]
            if sum>sum_min_1 and topk_values[2]>0 and (sum/topk_values[2]>2) and topk_values[2]!=0:
                final_location=max_location
                votes = sum
            elif topk_values[2]==0 and sum>sum_min_2+1:
                final_location=max_location
                votes = sum
            else:
                final_location='N'
                votes = 'N'            
        else:
            final_location='N'
            votes = 'N'
    return final_location, votes


def process_event(sample_number, random_matrix_numpy, event, reference_array_numpy, reference_array_comp_numpy, col, Threshold, sub_array_row, n_blocks, event_length):
    # transfer to cuda
    test_events_np = np.array(event, dtype=np.float32)

    # reference
    total_min_values_result =  count_below_threshold_numpy_v2(random_matrix_numpy, test_events_np, reference_array_numpy, col, Threshold, sub_array_row,n_blocks)
    final_location, votes = location_decide(sample_number, total_min_values_result, event_length)

    # recerence_complementary
    total_min_values_result =  count_below_threshold_numpy_v2(random_matrix_numpy, test_events_np, reference_array_comp_numpy, col, Threshold, sub_array_row,n_blocks)
    final_location_comp, votes_comp = location_decide(sample_number,total_min_values_result, event_length)

    # return final location
    return final_location, final_location_comp, votes, votes_comp