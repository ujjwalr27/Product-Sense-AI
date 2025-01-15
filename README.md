# ProductSense AI 🛍️ AI-Powered Shopping Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.4+-brightgreen.svg)](https://www.mongodb.com/)

SmartShop Insight is an intelligent product analysis platform that helps users make informed shopping decisions by leveraging advanced AI to analyze Amazon product reviews and answer natural language questions.

![SmartShop Insight Demo](static/plots/demo.png)

## ✨ Features

- 🤖 **AI-Powered Analysis**: Uses BERT and T5 models for accurate product insights
- 💬 **Natural Language Understanding**: Ask questions about products in plain English
- 📊 **Interactive Visualizations**: View sentiment analysis and word frequency patterns
- 📈 **Real-time Processing**: Instant analysis of thousands of product reviews
- 🔄 **History Tracking**: Keep track of your previous product analyses

## 🎯 Use Cases

1. **Product Research**
   - Analyze customer sentiments across reviews
   - Get quick answers about product features
   - Compare different product aspects

2. **Shopping Decisions**
   - Understand common pros and cons
   - Identify potential issues before purchase
   - Get insights from real user experiences

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd mini_proj_site
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and MongoDB URI
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

Visit `http://localhost:5000` in your browser to start using SmartShop Insight!

## 💡 Example Usage

1. Enter an Amazon ASIN (product ID) in the search box
2. Ask questions like:
   - "What do users say about battery life?"
   - "Are there any quality issues reported?"
   - "What are the main pros and cons?"
3. View generated visualizations and insights

![Analysis Example](static/plots/analysis_example.png)

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Database**: MongoDB
- **AI/ML**: 
  - PyTorch
  - Hugging Face Transformers (BERT, T5, DistilBERT)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **API**: Amazon Product API

## 📊 Project Structure

```
mini_proj_site/
├── app.py           # Flask application & routes
├── model.py         # AI models & analysis logic
├── requirements.txt # Project dependencies
├── static/         
│   └── plots/      # Generated visualizations
└── templates/      
    ├── index.html   # Main interface
    └── history.html # Analysis history
```

## 🔑 Environment Variables

Create a `.env` file with:
```
API_KEY=your_amazon_api_key
MONGO_URI=your_mongodb_uri
```

## 🔧 Troubleshooting

Common issues and solutions:

1. **API Key Issues**
   - Ensure your Amazon API key is valid
   - Check API rate limits
   - Verify .env file configuration

2. **MongoDB Connection**
   - Confirm MongoDB is running
   - Check connection string format
   - Verify network connectivity

3. **Model Loading**
   - Ensure sufficient RAM (4GB+ recommended)
   - Check for proper PyTorch installation
   - Verify internet connection for model downloads

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Author

- Utkarsh Kharche - [GitHub](https://github.com/utkarshkharche)

## 🙏 Acknowledgments

- Thanks to Hugging Face for their amazing transformer models
- Amazon Product API for real-time product data
- MongoDB for reliable data storage

## 📈 Future Enhancements

1. **Advanced Analytics**
   - Price trend analysis
   - Competitive product comparison
   - Review authenticity detection

2. **User Experience**
   - Mobile-responsive design
   - User preferences saving
   - Batch product analysis

3. **AI Capabilities**
   - Multi-language support
   - Image analysis integration
   - Price prediction modeling
