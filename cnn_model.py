import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


model = densenet121(weights=DenseNet121_Weights.DEFAULT)
model.eval()

nodes, _ = get_graph_node_names(model)
feature_extractor = create_feature_extractor(
    model, return_nodes=['adaptive_avg_pool2d'])


def transform_image(img):
    my_transforms = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return my_transforms(img).unsqueeze(0)


# kwargs were added for compatibility reasons
def extract_features(img, **kwargs):
    tensor = transform_image(img)
    out = feature_extractor(tensor)
    return list(out.values())[0].squeeze().detach().numpy()
