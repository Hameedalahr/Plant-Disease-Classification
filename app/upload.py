from huggingface_hub import upload_folder, upload_file

upload_file(
    path_or_fileobj ="app/trained_model/trained_model.h5",
    path_in_repo = "trained_model.h5",
    repo_id = "Hameedalahr/plants_trained_model",
    repo_type = "model"
)