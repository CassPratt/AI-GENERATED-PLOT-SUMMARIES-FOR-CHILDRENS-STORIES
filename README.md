# AI-GPSCS: AI-Generated Plot Summaries for Children's Stories
The main aim of the research is to develop an automated system that generates engaging and age-appropriate plot summaries for children's stories based on image descriptions. The research aims to leverage the fields of NLP, computer vision, and storytelling to create a novel pipeline that integrates image description generation and plot summaries generation.

## Repository structure
### Main branch
"main" branch contains all code used for the entire process separated by stages, they do not follow any specific order.
Folders:
- Datasets: Contains all images used for the project and notebooks to review them.
- Human evaluation results: Simple code to analyse evaluation results (images descriptions).
- Image captioning model: Code to fine-tune the image description model and analyse the results.
- Manual image captioning: Code to manual captioning the source images for training.
- Plot summaries generation: Comprehensive notebook with all steps for the final text generation.
- Story generation model: Code to test the text generation model.

### Web app code
Branch called "onlywebapp" contains the code for the system web app, when updated it is automatically deployed to an Azure Web App resource.

# Contributed by Cassandra Pratt Romero
