import torch

def join_chunks(input, chunk_size, mask, num_chunks_out_to_join):  # mask has size [B,num_chunks,num_chunks] input has size [B,F,T]

    """
    This function applies to a mel spectrogram of num_chunks*chunk_size frames a mixing mask of size
    [B,num_chunks_out,num_chunks_in] with num_chunks_in begin the # of chunks the input mel is divided in and
    num_chunks_out being the # of chunks the output will be divided in (usually half of num_chunks_in)
    params:
    input: the mel spectrogram of shape [B, features, frames]
    chunk_size: the # of frames in each chunk (f.e. 20)
    mask: the applied mixing mask of size [B,num_chunks_out,num_chunks_in]
    num_chunks_out_to_join: the No. of chunks which will be joined together at the output (must be equal to mask.size(1))
    """

    input_pred = []
    input_split = input.split(chunk_size, dim=2)  # [B, F, 1] # for every chunk

    # this is the manual implementation of the vector-matrix product of in chunks divied mel spec and mask
    for k_chunk in range(num_chunks_out_to_join):
        chunk_mask_product_sum = 0
        for idx, chunk in enumerate(input_split):
            mask_for_chunk = mask[:, k_chunk, idx]  # [B,1,1]
            mask_for_chunk = mask_for_chunk
            product = torch.zeros_like(chunk[:, :, :])
            for j in range(chunk.size(0)):
                product[j, :, :] = chunk[j, :, :].unsqueeze(0) * mask_for_chunk[j]

            chunk_mask_product_sum = chunk_mask_product_sum + product
        input_pred.append(chunk_mask_product_sum)
    input_pred = torch.cat(input_pred, dim=2)
    return input_pred


if __name__ == '__main__':

    #for testing
    a = torch.rand(2, 64, 1600)
    b = torch.rand(2, 2, 4)
    print(join_chunks(a, 400, b, 2).size())
