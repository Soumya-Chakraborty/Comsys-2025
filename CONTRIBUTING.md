# Contributing to ComsysHackathon

We welcome contributions to the ComsysHackathon project! This document provides guidelines for contributing to our computer vision solutions for gender classification and face recognition.

## 🤝 How to Contribute

### Types of Contributions

We appreciate all kinds of contributions:
- 🐛 **Bug reports and fixes**
- 🚀 **New features and enhancements**
- 📚 **Documentation improvements**
- 🧪 **Tests and benchmarks**
- 🎨 **Code quality improvements**
- 📊 **Dataset improvements**
- 🔧 **Performance optimizations**

## 🚀 Getting Started

### 1. Fork the Repository

```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ComsysHackathon.git
cd ComsysHackathon
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for both tasks
cd Task_A && pip install -r requirements.txt && cd ..
cd Task_B && pip install -r requirements.txt && cd ..

# Install development dependencies
pip install -r requirements-dev.txt  # If exists
```

### 3. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

## 📝 Development Guidelines

### Code Style

We follow Python best practices and PEP 8 standards:

```python
# Good example
class GenderClassifier:
    """A classifier for gender recognition using deep learning."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.model = self._load_model()
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Predict gender from input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with prediction probabilities
        """
        # Implementation here
        pass
```

### Code Quality Standards

1. **Type Hints**: Use type hints for function parameters and return values
2. **Docstrings**: All classes and functions must have docstrings
3. **Error Handling**: Implement proper exception handling
4. **Logging**: Use proper logging instead of print statements
5. **Constants**: Use constants for magic numbers and strings

### Testing

Before submitting, ensure your code passes all tests:

```bash
# Run existing tests
python Task_A/test_gpu_setup.py
python Task_B/test_installation.py

# Run your new tests
python -m pytest tests/ -v

# Check test coverage
python -m pytest --cov=Task_A --cov=Task_B tests/
```

### Performance Considerations

1. **Memory Efficiency**: Be mindful of memory usage, especially with large models
2. **GPU Utilization**: Optimize for GPU usage when available
3. **Batch Processing**: Implement efficient batch processing for inference
4. **Model Size**: Consider model compression techniques

## 🐛 Reporting Issues

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if the issue is already fixed
3. **Check documentation** for solutions

### Issue Template

When reporting bugs, please include:

```markdown
## Bug Description
A clear description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Run command '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.8.10]
- PyTorch version: [e.g., 2.0.0]
- CUDA version: [e.g., 11.8]
- GPU: [e.g., RTX 3080]

## Additional Context
Add any other context about the problem here.
```

## 🔧 Pull Request Process

### 1. Pre-submission Checklist

- [ ] Code follows the project's style guidelines
- [ ] All tests pass locally
- [ ] New code has appropriate tests
- [ ] Documentation is updated (if needed)
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch

### 2. Commit Message Format

Use clear, descriptive commit messages:

```bash
# Good examples
git commit -m "feat: add ensemble model support for Task A"
git commit -m "fix: resolve CUDA memory leak in face recognition"
git commit -m "docs: update installation instructions"
git commit -m "test: add unit tests for data augmentation"

# Format: type(scope): description
# Types: feat, fix, docs, style, refactor, test, chore
```

### 3. Pull Request Description

Use this template for your PR description:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
Describe the tests you ran and how to reproduce them.

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

## 📊 Task-Specific Guidelines

### Task A - Gender Classification

When contributing to gender classification:

1. **Bias Awareness**: Consider fairness and bias implications
2. **Data Balance**: Account for class imbalance in the dataset
3. **Evaluation Metrics**: Use appropriate fairness metrics
4. **Model Interpretability**: Provide explanations for predictions

```python
# Example: Adding a new bias metric
def demographic_parity_difference(y_true, y_pred, sensitive_attr):
    """Calculate demographic parity difference."""
    # Implementation with proper documentation
    pass
```

### Task B - Face Recognition

When contributing to face recognition:

1. **Privacy**: Ensure privacy-preserving techniques
2. **Robustness**: Test against various distortions
3. **Evaluation**: Use appropriate face recognition metrics
4. **Efficiency**: Optimize for real-time performance

```python
# Example: Adding a new distortion type
class NoiseAugmentation:
    """Add realistic noise to face images."""
    
    def __init__(self, noise_type: str = 'gaussian'):
        self.noise_type = noise_type
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Implementation here
        pass
```

## 🔬 Research Contributions

### Adding New Models

1. **Literature Review**: Cite relevant papers
2. **Baseline Comparison**: Compare against existing methods
3. **Ablation Studies**: Provide component analysis
4. **Reproducibility**: Include all necessary details

### Dataset Improvements

1. **Data Quality**: Ensure high-quality annotations
2. **Diversity**: Consider demographic diversity
3. **Ethics**: Follow ethical data collection practices
4. **Documentation**: Document data sources and preprocessing

## 📚 Documentation

### Code Documentation

```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4
) -> Dict[str, Any]:
    """Train a deep learning model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Dictionary containing training history and best model state
        
    Raises:
        ValueError: If model or data loaders are invalid
        RuntimeError: If CUDA is not available when expected
        
    Example:
        >>> model = GenderClassifier()
        >>> results = train_model(model, train_loader, val_loader)
        >>> print(f"Best accuracy: {results['best_accuracy']}")
    """
```

### README Updates

When adding new features, update relevant README sections:
- Installation instructions
- Usage examples
- Configuration options
- Performance benchmarks

## 🏆 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- Project documentation
- Academic publications (if applicable)

## 📞 Getting Help

If you need help with contributions:

1. **Check existing documentation** first
2. **Search closed issues** for similar problems
3. **Open a discussion** for questions
4. **Join our community** channels (if available)

## 🎯 Contribution Ideas

Looking for ways to contribute? Here are some ideas:

### Beginner-Friendly
- Fix typos in documentation
- Add more examples to README
- Improve error messages
- Add unit tests for existing functions

### Intermediate
- Implement new data augmentation techniques
- Add model compression methods
- Improve training visualization
- Add new evaluation metrics

### Advanced
- Implement new model architectures
- Add federated learning support
- Optimize inference speed
- Add multi-GPU training support

## 📋 Code Review Process

1. **Automated Checks**: PRs must pass all automated tests
2. **Peer Review**: At least one maintainer review required
3. **Documentation Review**: Ensure documentation is adequate
4. **Performance Review**: Check for performance regressions

## 🔒 Security

If you discover a security vulnerability:

1. **Do NOT open a public issue**
2. **Email the maintainers directly**
3. **Provide detailed information**
4. **Allow time for fix before disclosure**

## 📜 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 🙏 Thank You

Thank you for contributing to ComsysHackathon! Your contributions help advance computer vision research and make these tools available to the broader community.

---

**Questions?** Feel free to reach out via GitHub Issues or Discussions!