import torch # type: ignore

# #cov
# def count_below_threshold_cuda(random_matrix_tensor, test_events_tensor, reference_array_tensor, col, Threshold, sub_array_row,n_blocks):
#     """
#     Simulate the behaviour of Content Addressable Memory. Counts the occurrences of values below a specified threshold in a matrix transformation operation using CUDA.
    
#     Parameters:
#         random_matrix_tensor (Tensor): Random matrix for transformation.
#         test_events_tensor (Tensor): Test events to process.
#         reference_array_tensor (Tensor): Reference array for comparison.
#         col (int): Number of columns in the test events tensor.
#         Threshold (int): Threshold value to compare against.
#         sub_array_row (int): Row dimension for sub-arrays.
#         n_blocks (int): Number of blocks to process.
    
#     Returns:
#         Tensor: Sum of minimum values below the threshold for each block.
#     """
#     max_blocks = reference_array_tensor.shape[0] // sub_array_row
#     total_min_values = torch.zeros(max_blocks, dtype=torch.int,device='cuda')

#     for i in range(len(test_events_tensor) - col + 1):  # Adjusted to ensure we consider the last 'col' elements
#         read = test_events_tensor[i:i+col]
#         read = torch.matmul(read, random_matrix_tensor)
#         read = (read > 0).type(torch.int)
#         search = read - reference_array_tensor
        
#         abs_gt_tolerance_count = torch.sum(torch.abs(search) > 0, dim=1)

#         full_blocks = abs_gt_tolerance_count[:n_blocks * sub_array_row].view(n_blocks, sub_array_row)
#         block_mins = full_blocks.min(dim=1)[0]
#         block_min_values = (block_mins < Threshold).int()
#         total_min_values[:n_blocks] += block_min_values  

#     return total_min_values

def count_below_threshold_cuda_variation(gon, goff, random_matrix_tensor, test_events_tensor, reference_array_tensor, col, Threshold, sub_array_row,n_blocks, device):
    max_blocks = reference_array_tensor.shape[0] // sub_array_row
    total_min_values = torch.zeros(max_blocks, dtype=torch.int,device=device)

    unfolded_matrix = torch.stack([test_events_tensor[i:i+col].float() for i in range(len(test_events_tensor) - col + 1)])
    unfolded_result = torch.matmul(unfolded_matrix, random_matrix_tensor)
    unfolded_result = (unfolded_result > 0).type(torch.int8)


    for i in range(len(test_events_tensor) - col + 1):  # Adjusted to ensure we consider the last 'col' elements
    # for i in tqdm(range(len(test_events_tensor) - col + 1)):
        # read = test_events_tensor[i:i+col]
        # read = torch.matmul(read, random_matrix_tensor)
        # read = (read > 0).type(torch.int8)
        # search = torch.bitwise_xor(read, reference_array_tensor)
        # abs_gt_tolerance_count = torch.sum(search > 0, dim=1,dtype=torch.int8)

        read = unfolded_result[i]

        search = torch.bitwise_xor(read, reference_array_tensor)
        search_variation = torch.where(search == 1, gon, goff)
        abs_gt_tolerance_count = search_variation.sum(dim=1, dtype=torch.float16)

        full_blocks = abs_gt_tolerance_count[:n_blocks * sub_array_row].view(n_blocks, sub_array_row)
        block_mins = full_blocks.min(dim=1)[0]
        # print(block_mins)
        
        block_min_values = (block_mins < Threshold).int()

        total_min_values[:n_blocks] += block_min_values  
    return total_min_values  

def count_below_threshold_cuda(random_matrix_tensor, test_events_tensor, reference_array_tensor, col, Threshold, sub_array_row,n_blocks, device):
    max_blocks = reference_array_tensor.shape[0] // sub_array_row
    total_min_values = torch.zeros(max_blocks, dtype=torch.int,device=device)

    unfolded_matrix = torch.stack([test_events_tensor[i:i+col].float() for i in range(len(test_events_tensor) - col + 1)])
    unfolded_result = torch.matmul(unfolded_matrix, random_matrix_tensor)
    unfolded_result = (unfolded_result > 0).type(torch.int8)


    for i in range(len(test_events_tensor) - col + 1):  # Adjusted to ensure we consider the last 'col' elements
    # for i in tqdm(range(len(test_events_tensor) - col + 1)):
        # read = test_events_tensor[i:i+col]
        # read = torch.matmul(read, random_matrix_tensor)
        # read = (read > 0).type(torch.int8)
        # search = torch.bitwise_xor(read, reference_array_tensor)
        # abs_gt_tolerance_count = torch.sum(search > 0, dim=1,dtype=torch.int8)

        read = unfolded_result[i]

        search = torch.bitwise_xor(read, reference_array_tensor)
        abs_gt_tolerance_count = search.sum(dim=1, dtype=torch.uint8)

        
        # abs_gt_tolerance_count = (read[None, :] ^ reference_array_tensor).sum(dim=1)
        # abs_gt_tolerance_count = batch_hamming_distance(read, reference_array_tensor)
        

        full_blocks = abs_gt_tolerance_count[:n_blocks * sub_array_row].view(n_blocks, sub_array_row)
        block_mins = full_blocks.min(dim=1)[0]
        block_min_values = (block_mins < Threshold).int()
        total_min_values[:n_blocks] += block_min_values  

    return total_min_values  


def find_topk_locations(tensor, k=5):
    topk_values, topk_indices = torch.topk(tensor, k=k)
    return topk_values.tolist(), topk_indices.tolist()

def location_decide(sample_number, total_min_values_result, event_length):
    max_location = torch.argmax(total_min_values_result, dim=0).item()  
    topk_values, topk_indices = find_topk_locations(total_min_values_result, k=10)

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

def location_decide_contamination(sample_number, total_min_values_result, event_length):
    max_location = torch.argmax(total_min_values_result, dim=0).item()  
    topk_values, topk_indices = find_topk_locations(total_min_values_result, k=10)

    if sample_number >= 4000:
        sum_min_1 = 8
        sum_min_2 = 5
    else:
        sum_min_1 = 8/4000*sample_number
        sum_min_2 = 5/4000*sample_number
        # sum_min_1 = min(sum_min_1,4)
        # sum_min_2 = min(sum_min_2,3)

    if(total_min_values_result[max_location]>65):  
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
        elif topk_values[1]>0 and (topk_values[0]/topk_values[1]>2) and topk_values[0]>sum_min_2: #contamination
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

def process_event_variation(gon, goff, sample_number, random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device):
    # transfer to cuda
    test_events_tensor = torch.tensor(event, device=device).float()

    # reference
    total_min_values_result =  count_below_threshold_cuda_variation(gon, goff,random_matrix_tensor, test_events_tensor, reference_array_tensor, col, Threshold, sub_array_row,n_blocks, device)
    final_location, votes = location_decide(sample_number, total_min_values_result, event_length)

    # recerence_complementary
    total_min_values_result =  count_below_threshold_cuda_variation(gon, goff, random_matrix_tensor, test_events_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row,n_blocks, device)
    final_location_comp, votes_comp = location_decide(sample_number,total_min_values_result, event_length)

    # return final location
    return final_location, final_location_comp, votes, votes_comp

def process_event_contamination_variation(gon, goff, sample_number, random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device):
    # transfer to cuda
    test_events_tensor = torch.tensor(event, device=device).float()

    # reference
    total_min_values_result =  count_below_threshold_cuda_variation(gon, goff, random_matrix_tensor, test_events_tensor, reference_array_tensor, col, Threshold, sub_array_row,n_blocks, device)
    final_location, votes = location_decide_contamination(sample_number, total_min_values_result, event_length)

    # recerence_complementary
    total_min_values_result =  count_below_threshold_cuda_variation(gon, goff, random_matrix_tensor, test_events_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row,n_blocks, device)
    final_location_comp, votes_comp = location_decide_contamination(sample_number, total_min_values_result, event_length)

    # return final location
    return final_location, final_location_comp, votes, votes_comp

def process_event(sample_number, random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device):
    # transfer to cuda
    test_events_tensor = torch.tensor(event, device=device).float()

    # reference
    total_min_values_result =  count_below_threshold_cuda(random_matrix_tensor, test_events_tensor, reference_array_tensor, col, Threshold, sub_array_row,n_blocks, device)
    final_location, votes = location_decide(sample_number, total_min_values_result, event_length)

    # recerence_complementary
    total_min_values_result =  count_below_threshold_cuda(random_matrix_tensor, test_events_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row,n_blocks, device)
    final_location_comp, votes_comp = location_decide(sample_number,total_min_values_result, event_length)

    # return final location
    return final_location, final_location_comp, votes, votes_comp

def process_event_contamination(sample_number, random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device):
    # transfer to cuda
    test_events_tensor = torch.tensor(event, device=device).float()

    # reference
    total_min_values_result =  count_below_threshold_cuda(random_matrix_tensor, test_events_tensor, reference_array_tensor, col, Threshold, sub_array_row,n_blocks, device)
    final_location, votes = location_decide_contamination(sample_number, total_min_values_result, event_length)

    # recerence_complementary
    total_min_values_result =  count_below_threshold_cuda(random_matrix_tensor, test_events_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row,n_blocks, device)
    final_location_comp, votes_comp = location_decide_contamination(sample_number, total_min_values_result, event_length)

    # return final location
    return final_location, final_location_comp, votes, votes_comp