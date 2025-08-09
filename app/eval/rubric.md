# Evaluation Rubric for Multi-Agent Research Assistant

## Overview
This rubric defines how to evaluate the quality and performance of the research assistant across multiple dimensions.

## Scoring Dimensions

### 1. Faithfulness (0-1.0)
**Definition**: Proportion of answer content that is supported by retrieved evidence.

**Measurement**:
- Extract claims from the final answer
- Check each claim against retrieved contexts and citations
- Score = (supported_claims / total_claims)

**Thresholds**:
- 0.9-1.0: Excellent - All claims well-supported
- 0.7-0.89: Good - Most claims supported
- 0.5-0.69: Fair - Some unsupported claims
- <0.5: Poor - Many unsupported claims

### 2. Answerability (0-1.0)  
**Definition**: How well the system handles answerable vs unanswerable questions.

**Measurement**:
- For answerable questions: Did it provide a substantive answer?
- For unanswerable questions: Did it appropriately refuse or indicate limitations?
- Score = 1.0 if handled correctly, 0.0 if not

**Categories**:
- **Easy**: Well-established facts (capitals, basic science)
- **Medium**: Complex topics requiring synthesis
- **Hard**: Ambiguous or rapidly changing topics
- **Impossible**: Unanswerable questions (future events, personal opinions)

### 3. Citation Coverage (0-1.0)
**Definition**: Quality and completeness of source citations.

**Measurement**:
- Are major claims supported by citations?
- Are citation links resolvable and relevant?
- Are sources authoritative and recent when needed?

**Scoring**:
- Count major claims in answer
- Count claims with proper citations
- Check citation link validity
- Score = (properly_cited_claims / major_claims) × link_validity_rate

### 4. Completeness (0-1.0)
**Definition**: How thoroughly the answer addresses the question.

**Measurement**:
- Does it cover all aspects of the question?
- Are key perspectives included?
- Is depth appropriate for question complexity?

**Indicators**:
- Multiple relevant sources used
- Different viewpoints considered
- Appropriate level of detail

### 5. Coherence (0-1.0)
**Definition**: Logical flow and structure of the answer.

**Measurement**:
- Clear introduction and conclusion
- Logical progression of ideas
- Good use of structure (headings, bullets)
- Consistent tone and style

### 6. Currency (0-1.0)
**Definition**: Use of up-to-date information when relevant.

**Measurement**:
- For time-sensitive topics: Are recent sources used?
- Are outdated claims flagged or avoided?
- Is recency appropriate for the question type?

## Performance Benchmarks

### Overall Quality Grades
- **A (0.85-1.0)**: Production ready, high confidence
- **B (0.70-0.84)**: Good quality, minor issues
- **C (0.55-0.69)**: Acceptable, some concerns
- **D (0.40-0.54)**: Poor quality, major issues
- **F (<0.40)**: Unacceptable, system failure

### Expected Performance by Question Difficulty
| Difficulty | Target Faithfulness | Target Completeness | Target Confidence |
|------------|-------------------|--------------------|--------------------|
| Easy       | ≥0.9              | ≥0.8               | ≥0.8              |
| Medium     | ≥0.8              | ≥0.7               | ≥0.7              |
| Hard       | ≥0.7              | ≥0.6               | ≥0.6              |
| Impossible | N/A (should refuse)| N/A (should refuse)| ≤0.3              |

## Automated Evaluation Methods

### 1. Semantic Similarity
- Compare answer sentences with retrieved contexts
- Use embedding similarity (cosine similarity ≥ 0.7)
- Flag potential hallucinations

### 2. Citation Validation
- Extract URLs from citations
- Check HTTP response codes
- Verify title matches
- Check publication dates

### 3. Keyword Coverage
- Define expected keywords for each test question
- Check presence in final answer
- Weight by importance

### 4. Confidence Calibration
- Compare predicted confidence with actual performance
- Plot calibration curves
- Measure Brier score

## Manual Review Guidelines

### When to Manually Review
- Automated scores show inconsistency
- New question types or domains
- System confidence is low
- User feedback indicates issues

### Review Checklist
- [ ] Answer addresses the core question
- [ ] Claims are factually accurate
- [ ] Sources are credible and relevant
- [ ] No harmful or biased content
- [ ] Appropriate caveats provided
- [ ] Writing quality is professional

## Evaluation Workflow

1. **Run Automated Evaluation**
   - Execute test suite on evaluation dataset
   - Generate quantitative metrics
   - Flag outliers for manual review

2. **Manual Spot Checks**
   - Review 10% of results manually
   - Focus on edge cases and failures
   - Validate automated scoring

3. **Performance Analysis**
   - Compare across question categories
   - Identify improvement areas
   - Track performance over time

4. **Reporting**
   - Generate evaluation report
   - Include specific examples
   - Provide actionable recommendations

## Success Criteria

### Minimum Acceptable Performance
- Overall faithfulness: ≥0.7
- Citation coverage: ≥0.8
- Answerability: ≥0.9
- No harmful or misleading content

### Production Readiness Criteria
- Overall faithfulness: ≥0.85
- Citation coverage: ≥0.9
- Answerability: ≥0.95
- Latency: ≤30 seconds (95th percentile)
- Consistent performance across domains