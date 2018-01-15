import test_load_datas.load_data as load_data
import test_load_datas.tools as tools

data = load_data.pascalVOC(resize=True, resize_width=500, resize_height=400)
images, labels, boxes = data.get_next_batch(batch_size=50)
images, labels, boxes = data.get_next_batch(batch_size=50)
tool = tools.Tools(dataset_name='pascalVOC')

print(images[0].shape)
print(labels)
print(boxes)

n = 0
image = tool.draw_bounding_boxes(images[n], boxes[n], labels[n])
tool.show_image(image)
tool.save_image(image, filename='./save_test.jpg')