# 🤝 Contributing Guide

> Guidelines for contributing to Enterprise GPU Backtester v7.1

## 🎯 Welcome Contributors!

Thank you for your interest in contributing to the Enterprise GPU Backtester! This guide will help you get started with contributing code, documentation, bug reports, and feature requests.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## 📜 Code of Conduct

### Our Pledge
We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards
- **Be respectful** - Treat everyone with respect and kindness
- **Be collaborative** - Work together to achieve common goals
- **Be patient** - Help others learn and grow
- **Be constructive** - Provide helpful feedback and suggestions
- **Be professional** - Maintain professionalism in all interactions

### Enforcement
Project maintainers are responsible for clarifying standards and will take appropriate action in response to any instances of unacceptable behavior.

## 🚀 Getting Started

### Prerequisites
- Node.js 18.17+ (LTS recommended)
- Git 2.25+
- Docker (optional but recommended)
- HeavyDB access (for full testing)
- MySQL access (for full testing)

### Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/your-username/enterprise-gpu-backtester.git
cd enterprise-gpu-backtester
```

### Initial Setup
```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Set up pre-commit hooks
npm run prepare

# Run initial tests
npm run test

# Start development server
npm run dev
```

### Environment Configuration
Update `.env.local` with your development configuration:
```env
# Database connections for development
HEAVYDB_HOST=localhost
HEAVYDB_PORT=6274
MYSQL_HOST=localhost
MYSQL_PORT=3306

# Development settings
NODE_ENV=development
NEXTAUTH_SECRET=dev-secret-key
```

## 🔄 Development Workflow

### Branch Strategy
We use Git Flow with the following branch structure:

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Critical production fixes
- `release/*` - Release preparation

### Creating a Feature Branch
```bash
# Update develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... code changes ...

# Commit your changes
git add .
git commit -m "feat: add new feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### Commit Message Convention
We follow [Conventional Commits](https://conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat` - New features
- `fix` - Bug fixes
- `docs` - Documentation changes
- `style` - Code style changes (formatting, etc.)
- `refactor` - Code refactoring
- `test` - Adding or updating tests
- `chore` - Maintenance tasks
- `perf` - Performance improvements

**Examples:**
```bash
feat(strategies): add market regime detection algorithm
fix(api): resolve database connection timeout issues
docs(readme): update installation instructions
test(backtest): add comprehensive integration tests
```

## 📏 Code Standards

### TypeScript Guidelines
- Use **strict mode** TypeScript configuration
- Provide explicit types for all function parameters and return values
- Use interfaces for object types
- Prefer `const` over `let` when possible
- Use meaningful variable and function names

```typescript
// Good
interface BacktestResult {
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
}

const calculateSharpeRatio = (returns: number[], riskFreeRate: number): number => {
  // Implementation
}

// Avoid
const calc = (data: any): any => {
  // Implementation
}
```

### React Component Guidelines
- Use functional components with hooks
- Implement proper error boundaries
- Use TypeScript interfaces for props
- Follow component naming conventions

```typescript
// Component structure
interface BacktestResultsProps {
  backtest: Backtest
  onExport: (format: ExportFormat) => void
}

export const BacktestResults: React.FC<BacktestResultsProps> = ({
  backtest,
  onExport
}) => {
  // Component implementation
  return (
    <div className="backtest-results">
      {/* JSX content */}
    </div>
  )
}
```

### API Design Guidelines
- Follow RESTful principles
- Use appropriate HTTP status codes
- Implement proper error handling
- Add comprehensive input validation

```typescript
// API route structure
export async function GET(request: NextRequest) {
  try {
    // Validate request
    const { searchParams } = new URL(request.url)
    const schema = z.object({
      strategyId: z.string().uuid(),
      startDate: z.string().datetime(),
      endDate: z.string().datetime()
    })
    
    const params = schema.parse({
      strategyId: searchParams.get('strategyId'),
      startDate: searchParams.get('startDate'),
      endDate: searchParams.get('endDate')
    })
    
    // Business logic
    const result = await backtestService.create(params)
    
    return NextResponse.json(result, { status: 201 })
  } catch (error) {
    return handleApiError(error)
  }
}
```

### Database Guidelines
- Use parameterized queries to prevent SQL injection
- Implement proper connection pooling
- Add appropriate indexes
- Use transactions for data consistency

```typescript
// Database query example
const getStrategyResults = async (strategyId: string): Promise<BacktestResult[]> => {
  const query = `
    SELECT 
      total_return,
      sharpe_ratio,
      max_drawdown,
      created_at
    FROM backtest_results 
    WHERE strategy_id = $1 
    ORDER BY created_at DESC
  `
  
  return await db.query(query, [strategyId])
}
```

### CSS/Styling Guidelines
- Use Tailwind CSS utility classes
- Follow mobile-first responsive design
- Implement consistent spacing and typography
- Use CSS custom properties for theme variables

```tsx
// Styling example
export const TradingCard: React.FC<TradingCardProps> = ({ title, value, change }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
        {title}
      </h3>
      <div className="flex items-center justify-between">
        <span className="text-2xl font-bold text-gray-900 dark:text-white">
          {value}
        </span>
        <span className={`text-sm font-medium ${
          change >= 0 ? 'text-profit' : 'text-loss'
        }`}>
          {change >= 0 ? '+' : ''}{change.toFixed(2)}%
        </span>
      </div>
    </div>
  )
}
```

## 🧪 Testing Guidelines

### Testing Strategy
- **Unit Tests**: Test individual functions and components
- **Integration Tests**: Test component interactions and API endpoints
- **E2E Tests**: Test complete user workflows
- **Performance Tests**: Validate response times and load handling

### Writing Unit Tests
```typescript
// Component test example
import { render, screen, fireEvent } from '@testing-library/react'
import { BacktestButton } from '@/components/trading/BacktestButton'

describe('BacktestButton', () => {
  it('should start backtest when clicked', async () => {
    const onStart = jest.fn()
    
    render(
      <BacktestButton
        strategyId="strategy-123"
        onStart={onStart}
        disabled={false}
      />
    )
    
    const button = screen.getByRole('button', { name: /start backtest/i })
    fireEvent.click(button)
    
    expect(onStart).toHaveBeenCalledWith('strategy-123')
  })
  
  it('should display loading state during backtest', () => {
    render(
      <BacktestButton
        strategyId="strategy-123"
        onStart={jest.fn()}
        isLoading={true}
      />
    )
    
    expect(screen.getByText(/running.../i)).toBeInTheDocument()
  })
})
```

### API Testing
```typescript
// API test example
import { GET } from '@/app/api/strategies/route'
import { createMockRequest } from '@/tests/utils'

describe('/api/strategies', () => {
  it('should return strategies for authenticated user', async () => {
    const request = createMockRequest({
      method: 'GET',
      headers: { authorization: 'Bearer valid-token' }
    })
    
    const response = await GET(request)
    const data = await response.json()
    
    expect(response.status).toBe(200)
    expect(data.strategies).toHaveLength(2)
    expect(data.strategies[0]).toHaveProperty('id')
    expect(data.strategies[0]).toHaveProperty('name')
  })
  
  it('should return 401 for unauthenticated requests', async () => {
    const request = createMockRequest({ method: 'GET' })
    
    const response = await GET(request)
    
    expect(response.status).toBe(401)
  })
})
```

### E2E Testing
```typescript
// E2E test example
import { test, expect } from '@playwright/test'

test.describe('Strategy Management', () => {
  test('should create and run backtest for new strategy', async ({ page }) => {
    // Login
    await page.goto('/auth/login')
    await page.fill('[data-testid="email"]', 'test@example.com')
    await page.fill('[data-testid="password"]', 'password123')
    await page.click('[data-testid="login-button"]')
    
    // Navigate to strategies
    await page.goto('/strategies')
    
    // Create new strategy
    await page.click('[data-testid="create-strategy"]')
    await page.fill('[data-testid="strategy-name"]', 'E2E Test Strategy')
    await page.selectOption('[data-testid="strategy-type"]', 'TBS')
    await page.click('[data-testid="save-strategy"]')
    
    // Verify strategy was created
    await expect(page.locator('[data-testid="strategy-card"]')).toContainText('E2E Test Strategy')
    
    // Run backtest
    await page.click('[data-testid="run-backtest"]')
    await page.fill('[data-testid="start-date"]', '2024-01-01')
    await page.fill('[data-testid="end-date"]', '2024-01-31')
    await page.click('[data-testid="start-backtest"]')
    
    // Verify backtest is running
    await expect(page.locator('[data-testid="backtest-status"]')).toContainText('Running')
  })
})
```

### Test Commands
```bash
# Run all tests
npm run test

# Run tests with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e

# Run specific test file
npm test -- BacktestButton.test.tsx

# Run tests in watch mode
npm run test:watch
```

## 📚 Documentation

### Documentation Standards
- Write clear, concise documentation
- Include code examples for complex features
- Update documentation with code changes
- Use proper markdown formatting

### API Documentation
When adding new API endpoints, update the API documentation:

```typescript
/**
 * GET /api/strategies/:id/backtests
 * 
 * Get all backtests for a specific strategy
 * 
 * @param id - Strategy ID
 * @query page - Page number (default: 1)
 * @query limit - Items per page (default: 20)
 * 
 * @returns {BacktestSummary[]} List of backtest summaries
 * 
 * @example
 * ```
 * GET /api/strategies/123/backtests?page=1&limit=10
 * ```
 */
```

### Component Documentation
Document React components with JSDoc:

```typescript
/**
 * TradingChart component for displaying backtest results
 * 
 * @param data - Chart data points
 * @param type - Chart type ('line' | 'candlestick' | 'volume')
 * @param height - Chart height in pixels
 * @param onZoom - Callback when user zooms the chart
 * 
 * @example
 * ```tsx
 * <TradingChart
 *   data={backtestData}
 *   type="line"
 *   height={400}
 *   onZoom={(range) => console.log('Zoomed to:', range)}
 * />
 * ```
 */
```

## 🔍 Pull Request Process

### Before Submitting
1. **Update from develop**: Rebase your branch on the latest develop
2. **Run tests**: Ensure all tests pass
3. **Code quality**: Run linting and formatting
4. **Documentation**: Update relevant documentation
5. **Changelog**: Add entry to CHANGELOG.md if applicable

### PR Checklist
- [ ] Code follows project standards
- [ ] Tests added/updated for new functionality
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages follow convention
- [ ] PR title is descriptive
- [ ] Screenshots included for UI changes

### PR Template
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] E2E tests pass (if applicable)
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots for UI changes.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No console warnings/errors
```

### Review Process
1. **Automated Checks**: CI/CD pipeline validates code quality
2. **Peer Review**: At least one team member reviews the code
3. **Testing**: QA team validates functionality (for major features)
4. **Approval**: Project maintainer approves the PR
5. **Merge**: Squash and merge to develop branch

## 🐛 Issue Reporting

### Bug Reports
Use the bug report template:

```markdown
**Bug Description**
A clear description of what the bug is.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Screenshots**
If applicable, add screenshots.

**Environment**
- OS: [e.g. iOS]
- Browser: [e.g. chrome, safari]
- Version: [e.g. 22]

**Additional Context**
Any other context about the problem.
```

### Feature Requests
Use the feature request template:

```markdown
**Feature Description**
A clear description of what you want to happen.

**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
Describe the solution you'd like.

**Alternatives Considered**
Describe alternatives you've considered.

**Additional Context**
Any other context or screenshots.
```

### Security Issues
For security vulnerabilities:
1. **DO NOT** create a public issue
2. Email security@your-domain.com
3. Include detailed description
4. Allow time for investigation before disclosure

## 👥 Community

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Slack**: Real-time communication (invite-only)
- **Email**: security@your-domain.com for security issues

### Getting Help
- Check existing documentation
- Search closed issues for similar problems
- Ask questions in GitHub Discussions
- Join our Slack community for real-time help

### Recognition
Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Annual contributor spotlight
- Conference speaking opportunities

## 🏆 Contribution Levels

### 🌱 First-time Contributors
- Good first issues labeled with `good-first-issue`
- Documentation improvements
- Test coverage improvements
- Bug fixes with clear reproduction steps

### 🌿 Regular Contributors
- Feature implementations
- Performance optimizations
- Complex bug fixes
- Code reviews

### 🌳 Core Contributors
- Architecture decisions
- Release management
- Mentoring new contributors
- Security reviews

## 🎓 Learning Resources

### Technology Stack
- [Next.js Documentation](https://nextjs.org/docs)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Jest Testing Framework](https://jestjs.io/docs/getting-started)
- [Playwright E2E Testing](https://playwright.dev/)

### Financial Trading
- [Options Trading Basics](https://example.com)
- [Backtesting Strategies](https://example.com)
- [Risk Management](https://example.com)

### Development Best Practices
- [Clean Code Principles](https://example.com)
- [Git Flow Workflow](https://example.com)
- [API Design Guidelines](https://example.com)

## 📈 Metrics and KPIs

We track contribution metrics to improve the development process:

- **Pull Request Metrics**: Time to review, merge rate
- **Issue Metrics**: Time to resolve, customer satisfaction
- **Code Quality**: Test coverage, bug rates
- **Performance**: API response times, page load speeds

## 🎯 Roadmap Participation

Contributors can participate in roadmap planning:
- Quarterly planning sessions
- Feature voting
- Architecture discussions
- User feedback sessions

## 🏅 Recognition Program

### Monthly Recognition
- **Bug Hunter**: Most bugs reported/fixed
- **Feature Champion**: Best feature implementation
- **Documentation Hero**: Best documentation contributions
- **Community Helper**: Most helpful in discussions

### Annual Awards
- **Outstanding Contributor**: Overall contribution excellence
- **Innovation Award**: Most innovative feature
- **Quality Champion**: Highest code quality standards
- **Mentor of the Year**: Best mentoring of new contributors

---

## 🙏 Thank You

Thank you for contributing to the Enterprise GPU Backtester! Your contributions help make this project better for everyone in the trading and financial technology community.

**Questions?** Reach out to the maintainers or ask in GitHub Discussions.

**Happy coding!** 🚀