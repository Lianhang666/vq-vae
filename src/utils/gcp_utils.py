from google.cloud import storage

def upload_blob_from_memory(bucket_name, blob_data, destination_blob_name, content_type):
    """Uploads a file to the bucket from memory."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_file(blob_data, content_type=content_type)

    print(f"Data uploaded to {destination_blob_name}.")


def save_history_to_gcs(bucket_name, history_json, destination_blob_name):
    client = storage.Client()
    # bucket_name = 'experiment_results_23_5'
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(history_json, content_type='application/json')
    print("History object saved to GCS")