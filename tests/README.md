# Testing Guide

This directory contains comprehensive tests for the CrediLinq Content Agent application.

## 🛡️ Database Safety

**IMPORTANT**: All tests use mocked database connections. **NO actual database operations are performed** during testing. Your production database is completely safe.

## Test Structure

```
tests/
├── api/                    # API endpoint tests
│   ├── test_blogs.py      # Blog CRUD operations
│   └── test_campaigns.py  # Campaign management
├── integration/            # End-to-end workflow tests
│   └── test_campaign_workflow.py
├── unit/                   # Unit tests
│   └── test_database.py   # Database operation logic (mocked)
├── conftest.py            # Test configuration and fixtures
└── README.md              # This file
```

## Running Tests

### Backend Tests (Python)

```bash
# All backend tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/ -m unit -v

# Integration tests only  
python -m pytest tests/ -m integration -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Frontend Tests (TypeScript)

```bash
cd frontend/

# All frontend tests
npm run test

# With coverage
npm run test:coverage

# Interactive UI
npm run test:ui
```

### All Tests

```bash
# Run everything (backend + frontend + linting)
python scripts/test_runner.py --all

# Backend only
python scripts/test_runner.py --backend-only

# Frontend only  
python scripts/test_runner.py --frontend-only
```

## Test Categories

### 🔧 **API Tests** (`tests/api/`)
- Campaign CRUD operations
- Blog post management
- Task status updates
- Error handling and validation
- Authentication and permissions

### 🔄 **Integration Tests** (`tests/integration/`)
- Complete campaign workflows
- Blog-to-campaign creation flow
- Task execution pipelines
- End-to-end user journeys

### 🧪 **Unit Tests** (`tests/unit/`)
- Database query logic (mocked)
- Individual function behavior
- Component rendering
- Hook functionality

### ⚛️ **Frontend Tests** (`frontend/src/`)
- Component rendering and interactions
- Hook state management
- User event handling
- API integration (mocked)

## Test Fixtures

Common test data is provided through fixtures in `conftest.py`:

- `sample_blog_post`: Mock blog post data
- `sample_campaign`: Mock campaign data  
- `sample_campaign_task`: Mock task data
- `mock_db_connection`: Mocked database connection
- `test_client`: FastAPI test client

## Mocking Strategy

### Database Mocking
All database operations are mocked using `unittest.mock`:
- No real database connections
- Predictable test data
- Isolated test execution
- Safe for CI/CD pipelines

### API Mocking (Frontend)
Frontend tests use MSW (Mock Service Worker):
- Realistic API responses
- Network error simulation
- Request/response validation

## Writing New Tests

### Backend Test Example

```python
def test_new_feature(test_client, mock_db_config):
    mock_config, mock_conn, mock_cursor = mock_db_config
    mock_cursor.fetchone.return_value = ("expected-result",)
    
    response = test_client.post("/api/endpoint", json={"data": "test"})
    
    assert response.status_code == 200
    assert response.json()["result"] == "expected-result"
```

### Frontend Test Example

```typescript
it('should handle user interaction', async () => {
  const user = userEvent.setup()
  render(<MyComponent />)
  
  await user.click(screen.getByRole('button'))
  
  expect(screen.getByText('Expected Result')).toBeInTheDocument()
})
```

## Best Practices

1. **Always mock database operations** - Never use real database connections
2. **Test error conditions** - Don't just test the happy path
3. **Use descriptive test names** - Test names should explain what is being tested
4. **Keep tests isolated** - Each test should be independent
5. **Mock external dependencies** - API calls, file systems, databases
6. **Test user behavior** - Focus on what users actually do

## Coverage Goals

- **API Endpoints**: 90%+ coverage
- **Core Business Logic**: 95%+ coverage  
- **Database Operations**: 85%+ coverage (logic only)
- **Frontend Components**: 80%+ coverage
- **Hooks and Utilities**: 90%+ coverage

## Continuous Integration

Tests run automatically on:
- Every commit (pre-commit hook)
- Pull request creation
- Deployment pipeline
- Nightly builds

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **Mock Failures**: Check that mocks are properly configured in `conftest.py`
3. **Async Test Issues**: Use `pytest-asyncio` for async test functions
4. **Frontend Test Failures**: Check MSW mock server configuration

### Getting Help

- Check existing tests for examples
- Review `conftest.py` for available fixtures
- Run tests with `-v` flag for verbose output
- Use `pytest --tb=short` for concise error messages

---

## ✅ Test Summary

This test suite provides comprehensive coverage of:
- ✅ All API endpoints (campaigns, blogs, analytics)
- ✅ Complete user workflows (blog creation → campaign → distribution)
- ✅ Database operations (100% mocked, zero DB impact)
- ✅ Frontend components and user interactions
- ✅ Error handling and edge cases
- ✅ Performance and reliability scenarios

**Remember**: All database operations are mocked. Your production data is completely safe! 🛡️