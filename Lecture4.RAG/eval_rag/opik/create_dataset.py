import opik

client = opik.Opik()
suite = client.get_or_create_test_suite(
    name="simple_test_dataset_v4", project_name="Test evaluation")
