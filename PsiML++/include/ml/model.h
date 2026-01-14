#pragma once

#include "../core/types.h"
#include "../core/config.h"
#include "../core/exception.h"
#include "../math/vector.h"
#include "../math/matrix.h"
#include <string>

namespace psi {
    namespace ml {

        // Model state
        enum class ModelState : core::u8 {
            Untrained = 0,
            Training = 1,
            Trained = 2
        };

        // Base model interface
        template<typename T>
        class Model {
        public:
            virtual ~Model() = default;

            // Training
            virtual void fit(const math::Matrix<T>& X, const math::Vector<T>& y) = 0;

            // Prediction
            virtual math::Vector<T> predict(const math::Matrix<T>& X) const = 0;

            // Single sample prediction
            virtual T predict_single(const math::Vector<T>& x) const {
                math::Matrix<T> X(1, x.size());
                for (core::usize i = 0; i < x.size(); ++i) {
                    X(0, i) = x[i];
                }
                return predict(X)[0];
            }

            // Model state
            PSI_NODISCARD ModelState state() const noexcept { return state_; }
            PSI_NODISCARD bool is_trained() const noexcept { return state_ == ModelState::Trained; }

            // Model name
            PSI_NODISCARD virtual std::string name() const = 0;

        protected:
            ModelState state_ = ModelState::Untrained;
        };

        // Supervised model with score method
        template<typename T>
        class SupervisedModel : public Model<T> {
        public:
            // Score (R^2 for regression, accuracy for classification)
            virtual T score(const math::Matrix<T>& X, const math::Vector<T>& y) const = 0;
        };

        // Unsupervised model interface (no labels)
        template<typename T>
        class UnsupervisedModel {
        public:
            virtual ~UnsupervisedModel() = default;

            // Fit without labels
            virtual void fit(const math::Matrix<T>& X) = 0;

            // Transform data
            virtual math::Matrix<T> transform(const math::Matrix<T>& X) const = 0;

            // Fit and transform
            virtual math::Matrix<T> fit_transform(const math::Matrix<T>& X) {
                fit(X);
                return transform(X);
            }

            // Model state
            PSI_NODISCARD ModelState state() const noexcept { return state_; }
            PSI_NODISCARD bool is_fitted() const noexcept { return state_ == ModelState::Trained; }

            // Model name
            PSI_NODISCARD virtual std::string name() const = 0;

        protected:
            ModelState state_ = ModelState::Untrained;
        };

        // Clustering model interface
        template<typename T>
        class ClusteringModel : public UnsupervisedModel<T> {
        public:
            // Predict cluster labels
            virtual math::Vector<core::i32> predict(const math::Matrix<T>& X) const = 0;

            // Fit and predict
            virtual math::Vector<core::i32> fit_predict(const math::Matrix<T>& X) {
                this->fit(X);
                return predict(X);
            }
        };

    } // namespace ml
} // namespace psi
