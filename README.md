# FitBuddy

FitBuddy is an AI-powered chatbot designed to provide personalized fitness assistance. By combining BERT-based intent classification with GPT-2 response generation, FitBuddy delivers tailored fitness advice and enhances user engagement.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview

FitBuddy leverages the power of modern natural language processing models to understand and respond to user queries related to fitness. It utilizes:
- **BERT** for intent classification.(Model link -> https://www.kaggle.com/models/sankeerthnadu/bert-intent-model/Transformers/default/1 )
- **GPT-2** for generating contextually relevant responses.(Model link -> https://www.kaggle.com/models/sankeerthnadu/gpt-2-transformer-based-model/Transformers/default/1 )

This combination allows FitBuddy to not only interpret a wide range of fitness-related queries but also deliver responses that are coherent, relevant, and personalized.

## Features

- **Intent Classification:** Uses BERT to accurately determine the user's fitness intent.
- **Personalized Response Generation:** GPT-2 generates responses tailored to the specific needs and goals of the user.
- **User Interaction Flow:** 
  - User inputs a query.
  - BERT classifies the intent.
  - GPT-2 generates a response based on the confirmed intent.
- **Performance Evaluation:** Experimental comparisons demonstrate improvements in response relevance, user satisfaction, and overall interaction quality.

## Architecture

The project integrates two primary models:
1. **Intent Classification with BERT:** Fine-tuned on a dataset of fitness-related queries to predict user intent.
2. **Response Generation with GPT-2:** Generates human-like responses based on the classified intent and the original query.

**Workflow:**
1. **User Input:** A query is provided by the user.
2. **Intent Analysis:** The query is processed by BERT to determine the fitness-related intent.
3. **Prompt Formation:** The identified intent and query are combined to create a prompt.
4. **Response Generation:** GPT-2 generates a response that is both contextually relevant and aligned with the user's intent.

## Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/FitBuddy.git
   cd FitBuddy

2. **Install Dependencies: Ensure you have Python 3.7+ installed. Then, install the necessary Python packages:**
   ```bash
   pip install -r requirements.txt
## Usage

1. Training:

- Prepare your dataset of fitness-related queries with intent labels.
- Fine-tune the BERT model on your dataset.
  Example command:
  ```bash
  python train_intent_classifier.py --data data/fitness_dataset.csv --epochs 5

2. Running the Chatbot:

Start the chatbot server or run the script that initiates the interactive session.
- Example command:
```bash
python run_chatbot.py
```

3.Interacting with FitBuddy:

- Input your fitness-related query.
- Follow on-screen instructions to confirm intent (if applicable).
- Receive a tailored response based on your query.

## Experimental Results
**The project includes a comparative study of model performance:**

- Response Relevance: 15% improvement with intent classification.
- User Satisfaction: 20% increase with personalized responses.
- Contextual Accuracy: Notable enhancement in the accuracy and fluency of responses.
- These results underline the benefits of incorporating intent classification into fitness guidance chatbots.

## Future Work

- Expand the range of intent categories.
- Incorporate user history to further personalize advice.
- Optimize model performance for real-time interactions.
  
## Contributing
**Contributions are welcome! Please follow these steps:**

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Commit your changes and create a pull request.
- Ensure your code adheres to the project’s coding standards.

## License
This project is licensed under the MIT License.

## 📬 Contact
For any inquiries, reach out at [sankeerth2004@gmail.com] or connect on LinkedIn www.linkedin.com/in/lucky-luc28.
