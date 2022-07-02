import json


def json_reader(jsonfile):
    with open(jsonfile, 'r') as f:
        dict = json.load(f)
    return dict

if __name__ == '__main__':
    d1 = '/home/tu-wan/second2t/project/video_captioning/grounded-video-description/data/anet/anet_entities_test_1.json'
    d2 = '/home/tu-wan/second2t/project/video_captioning/grounded-video-description/data/anet/anet_entities_test_2.json'
    d = '/home/tu-wan/second2t/project/video_captioning/grounded-video-description/data/anet/anet_entities_test.json'
    d1_d = json_reader(d1)
    d2_d = json_reader(d2)
    tmp1 =d1_d['v_bXdq2zI1Ms0']
    tmp2 =d2_d['v_bXdq2zI1Ms0']
    gts = []
    filenames = [d1,d2]
    # self.n_ref_vids = set()
    for filename in filenames:
        gt = json.load(open(filename))
        # self.n_ref_vids.update(gt.keys())
        gts.append(gt)