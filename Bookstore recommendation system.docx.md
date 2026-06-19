**BOOK-LOG**  
**THE BOOKSTORE RECOMMENDATION SYSTEM**

Team name: The Nonchalants

Leader: Vo Van Khanh (24ITE048)  
Members: Truong Minh Nhat Khanh (24ITE047)

| Goal | Improve book discovery by providing personalized, explainable reading suggestions for existing and new readers. |
| :---: | :---- |
| Domain | E-commerce / Online Bookstore. |
| Recommendation method | Dynamic weighted hybrid system combining Collaborative Filtering, Content-Based Filtering and a Bayesian popularity fallback. |
| Data collection | [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset): `Books.csv` contains ISBN, title, author, publisher, publication year and cover-image URLs; `Ratings.csv` contains ratings on a 0-10 scale; `Users.csv` contains user location and age. A filtered `data/processed` subset is used to reduce training time and memory usage. |
| Runtime storage | On first launch, the cleaned CSV data is imported into SQLite. The Flask website reads books, users and ratings from SQLite and stores new or updated ratings there. |
| Evaluation results | On `data/processed`: RMSE = **1.4551**, MAE = **1.1050**, HitRate@10 = **0.0500**, Precision@10 = **0.0050**. |
| Verification | The automated test suite contains **11 passing tests**, covering dataset cleaning, recommendation behavior, the Flask interface and SQLite persistence. |
| Conclusion | The implemented system combines metadata similarity, matrix-factorization predictions and a popularity fallback. It supports existing users, favorite-title cold start and persistent rating updates, but the Top-N ranking still requires improvement. |
| Online repository | Not published yet. |
| Team R\&R Peer review | Pending team confirmation before submission. |
| History of team meeting | Pending team confirmation before submission. |

**1\. Introduction**

Background: The exponential growth of e-commerce has led to severe information overload for consumers. In the literary domain, users often struggle to find relevant reading material among millions of available titles.  
Motivation:  
\+ Alleviate the common "decision fatigue" associated with navigating massive book catalogs while actively combating poor purchasing choices.  
\+ Overcoming information overload: Too much information can confuse users and slow decision-making.   
\+ Enhancing Customer Personalization and Retention: By alleviating decision fatigue amidst millions of titles, the personalized system optimizes user experience, thereby increasing return rates and ensuring sustainable customer retention.  
\+ Driving Revenue and Sales: Better user experience, personalization, and simplified decisions lead to higher conversion rates, increased purchases, and ultimately more revenue.  
What is the research issue?  
\+ **Core pain:** Readers have access to millions of books but struggle to decide what they should read next.  
\+ A recommendation method based only on previous interactions may repeatedly suggest familiar items, while a method based only on book metadata cannot learn community-level preferences.  
\-\> The core research issue is how to combine a user's explicit rating history with the limited metadata available in the dataset so that the system can produce useful Top-N recommendations and still support users with little or no rating history.  
Why is this research important?  
\+ Combining historical preferences with metadata-based discovery can reduce choice paralysis and improve the relevance of recommendations.  
\+ A personalized ranking can also give less-popular books an opportunity to reach readers whose profiles are similar to those books.  
 \+ By providing an automated tool that suggests what to read based on the user's explicit rating history or an entered favorite book, we aim to:

* Help users optimize their reading budget.  
* Reduce the time wasted on endless scrolling.  
* Make book exploration a more seamless, diverse, intellectually rewarding and enjoyable experience.  
* Make the suggested titles more suitable.

**2\. Problem description and modeling**

Problem modeling  
\+ **The User-Item Matrix:** The core environment is modeled as a bipartite network mapping a set of Users (U) to a set of Books (B). We construct a User-Item interaction matrix R, where each known entry rui is an explicit rating. Although the source dataset uses a 0-10 scale, the current implementation trains on ratings from **1 to 10**; zero ratings and implicit signals are excluded from model training.  
\+ **The Sparsity Challenge:** Because each user interacts with only a small fraction of the catalog, the matrix R is highly sparse. In the processed dataset, 12,118 known ratings across 234 users and 1,500 books leave approximately **96.5%** of the user-item matrix empty.  
\+ **The Objective:** The Collaborative Filtering branch is framed as a **Matrix Completion Task**, learning a prediction function f(u, i) → rui for unknown user-book pairs. The overall system then combines these predictions with content and popularity scores to generate a ranked Top-N list of unread books.  
Mathematical modeling of user feedback  
\+ To process user feedback, we employ **Matrix Factorization (MF)** with Baseline Predictors. This algorithm decomposes the sparse matrix R into two lower-dimensional, dense matrices: a User latent feature matrix P and a Book latent feature matrix Q.  
\+ **Prediction Formula:** The predicted rating rui is mathematically formulated by combining global baseline shifts with the dot product of the learned latent vectors:  
![][image1]  
*Where:*

* μ: The global average of all explicit training ratings.  
* bu, bi: The learned user bias and book bias.  
* Pu, Qi: The learned latent-factor vectors of user u and book i.

**\+ Optimization & Loss Function:** The parameters are trained with Stochastic Gradient Descent (SGD) by minimizing Regularized Squared Error. The L2 regularization coefficient λ penalizes large bias and latent-factor values to reduce overfitting:

**![][image2]**

*(Note: K is the set of known explicit user-book ratings used for training.)*

**3\. Recommendation by your idea**

What is your idea for recommendation?    
\+  **A Dynamic Weighted Hybrid System:** To overcome the limitations of individual algorithms, the core idea is to engineer a hybrid architecture that processes data through two parallel pipelines before fusing the results.  
\+ **Pipeline 1: Content-Based Filtering:** This branch uses TF-IDF (Term Frequency-Inverse Document Frequency) on the metadata available in `Books.csv`: **book title, author, publisher and publication year**. It computes cosine similarity between books and builds a content-preference profile from books that a user rated highly. The current dataset does not contain synopsis or genre fields.  
\+ **Pipeline 2: Collaborative Filtering (MF):** As modeled in Section 2, this branch uses Biased Matrix Factorization to learn community preferences from historical **explicit ratings**.  
\+ **The Fusion Mechanism:** The final recommendation score is calculated via a dynamically weighted ensemble:  
![][image3]  
The implemented fusion formula is:

`alpha = rating_count / (rating_count + 10)`

`final_score = 0.9 × (alpha × CF_score + (1 - alpha) × Content_score) + 0.1 × Popularity_score`

The parameter alpha dynamically adjusts based on the user's profile density. For a brand-new user, alpha is 0, so recommendations rely on the entered favorite book and popularity fallback. As the number of ratings increases, alpha increases and shifts more weight toward Collaborative Filtering. All component scores are min-max normalized before fusion.  
What kind of recommendation can be generated?  
\+  **Personalized "Top Picks For You":** The primary output is a customized, ranked list generated after selecting an existing reader profile. The system scores unread books and returns the Top-N books with the highest hybrid utility.  
\+ **Metadata-Based "Item-to-Item" Suggestions:** The backend and CLI can generate similar-book recommendations using TF-IDF cosine similarity over title, author, publisher and publication year. The current web page focuses on catalog search and personalized Top-N recommendations; a dedicated book-detail page is future work.  
\+ **Adaptive Cold-Start Onboarding:** A new reader enters a favorite book. The system builds a content profile from that title and combines it with a small popularity fallback without requiring previous ratings.

**4\. Implementation**

**Development environment**

* Programming language: Python 3.12+.
* Data processing: pandas, NumPy and SciPy.
* Machine learning: scikit-learn.
* Web backend: Flask.
* Runtime database: SQLite through Python's built-in `sqlite3` module.
* User interface: HTML and CSS.
* Testing: pytest.
* Dataset: Kaggle Book Recommendation Dataset. The current web-ready `data/processed` subset and generated `data/booklog.db` contain **1,500 books**, **234 users** and **12,118 ratings**.

**System architecture**

```text
Books.csv / Ratings.csv / Users.csv
                  |
          Data cleaning and filtering
                  |
         SQLite: books / users / ratings
             /                  \
 TF-IDF content model      Biased Matrix Factorization
             \                  /
        Dynamic weighted hybrid + popularity fallback
                         |
                  Unread Top-N books
                         |
               Flask bookstore website
                         |
             Save or update reader rating
                         |
                       SQLite
```

The Flask website follows the SQLite runtime path shown above. The CLI reads a selected CSV dataset directly for offline recommendation and evaluation.

**Main components**

* `src/booklog/data.py`: Loads the three CSV files, validates columns, cleans invalid years and ratings, removes duplicates and filters sparse interactions.
* `src/booklog/database.py`: Creates the SQLite schema, imports the initial CSV dataset, loads runtime data and upserts user ratings.
* `src/booklog/content.py`: Creates TF-IDF vectors from title, author, publisher and publication year; computes cosine similarity and user content profiles.
* `src/booklog/collaborative.py`: Implements the prediction formula shown in Image 1 and optimizes the regularized loss shown in Image 2 using SGD.
* `src/booklog/hybrid.py`: Normalizes and combines content, collaborative and popularity scores; removes books already rated by the user.
* `src/booklog/evaluation.py`: Performs leave-one-out evaluation and calculates RMSE, MAE, HitRate@K and Precision@K.
* `app.py`: Initializes SQLite from the processed CSV files when necessary, loads runtime data, trains and caches the model, handles rating updates and serves the bookstore interface.

**Database design**

The SQLite database contains three related tables:

* `books`: Stores ISBN, title, author, publication year, publisher and cover-image URLs.
* `users`: Stores reader ID, location and age.
* `ratings`: Stores one rating per user-book pair using a composite primary key. Foreign keys reference `users` and `books`.

When an existing reader submits a rating from the website, the application inserts or updates the corresponding `ratings` row. It then clears the model cache so that the next recommendation request trains with the updated data.

**User interface**

The simple bookstore web interface provides:

* A responsive catalog with book-cover images from `Image-URL-M`.
* Search by book title or author.
* Existing-reader selection for personalized Top-N recommendations.
* New-reader cold-start recommendation from one entered favorite book.
* Match percentages for hybrid recommendations.
* A rating form that lets a selected existing reader save or update a score from 1 to 10.

Run the application:

```powershell
python app.py
```

Then open `http://127.0.0.1:5000`.

The implementation is verified with:

```powershell
python -m pytest -q
```

The current automated test result is **11 passed**.

**5\. Evaluation and discussion**

**How was the evaluation designed?**

* Only explicit ratings greater than zero are used.
* Users must have at least three explicit ratings to participate.
* Leave-one-out evaluation holds one rating from each eligible user as test data and trains the model on the remaining ratings.
* RMSE and MAE measure Collaborative Filtering rating-prediction error.
* A held-out rating of at least 7/10 is considered relevant.
* HitRate@10 measures how often the relevant held-out book appears in the hybrid Top-10 list.
* Precision@10 measures the number of successful hits divided by all recommendation positions.

The evaluation command was:

```powershell
python cli.py --data-dir data/processed evaluate --top-k 10 --max-users 100
```

**Evaluation results**

| Metric | Result |
| :--- | ---: |
| RMSE | 1.4551 |
| MAE | 1.1050 |
| HitRate@10 | 0.0500 |
| Precision@10 | 0.0050 |
| Evaluated ratings | 234 |
| Evaluated relevant users | 100 |

**Discussion**

* The RMSE and MAE results show that Matrix Factorization learns useful rating patterns, but prediction errors remain because the interaction matrix is sparse and the processed dataset is intentionally small.
* HitRate@10 and Precision@10 are low. This indicates that the current Top-N ranking requires improvement before production use.
* Content-Based Filtering helps cold-start users and produces understandable recommendations, but its ability to distinguish themes is limited because the dataset has no synopsis or genre fields.
* The current model does not use age or location from `Users.csv`; these attributes could support future demographic-aware recommendations.
* Dynamic alpha prevents the model from relying too strongly on Collaborative Filtering for users with little history.
* The 10% Bayesian popularity fallback stabilizes recommendations when a content profile is empty or weak.
* The processed subset improves web performance but removes many users, books and interactions from the original Kaggle dataset. This creates a trade-off between response time and recommendation quality.
* SQLite provides persistent runtime storage and allows ratings submitted from the website to be reused by later recommendation requests. However, the current application retrains the in-memory model after a rating update instead of performing incremental learning.
* Future evaluation should include coverage, diversity, novelty and online A/B testing.

**6\. Conclusion**

**What is our insight?**

A single recommendation method cannot effectively address personalization, sparsity and cold-start at the same time. Content-Based Filtering can recommend from a favorite book even when no rating history exists, while Matrix Factorization learns hidden preferences from the community. Dynamic weighting allows the system to move gradually from content-based exploration toward collaborative personalization.

**What did we experience and realize?**

* Data cleaning and interaction filtering strongly affect recommendation quality and training time.
* A fixed hybrid weight is not appropriate for every user. New and active users require different balances.
* Rating-prediction accuracy alone is insufficient; Top-N ranking metrics must also be measured.
* Using a smaller processed dataset makes the website practical on a local computer, but reduces the available collaborative information.
* Separating initial CSV preparation from SQLite runtime storage makes the website able to persist user feedback without modifying the source dataset.
* The most valuable future improvements are adding genre and synopsis metadata, supporting authentication and multiple favorite books, tuning hyperparameters on a validation set, and adding item-to-item suggestions to the web interface.

**7\. References**

1. Arash Nic, *Book Recommendation Dataset*, Kaggle, CC0 1.0: <https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset>
2. Yehuda Koren, Robert Bell and Chris Volinsky, *Matrix Factorization Techniques for Recommender Systems*, IEEE Computer, 2009.
3. Francesco Ricci, Lior Rokach and Bracha Shapira, *Recommender Systems Handbook*, Springer.
4. scikit-learn documentation, *TfidfVectorizer*: <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>
5. scikit-learn documentation, *Cosine Similarity*: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html>
6. Flask documentation: <https://flask.palletsprojects.com/>
7. SQLite documentation: <https://www.sqlite.org/docs.html>

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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>
