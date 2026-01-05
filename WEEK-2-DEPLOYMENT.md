# Week 2: Deployment Checklist

Use this checklist to prepare Week 2 code for production.

## Infrastructure Setup

### Weaviate Database
- [ ] Weaviate instance running (Docker, cloud, or self-hosted)
  ```bash
  docker run -d -p 8080:8080 -p 50051:50051 \
    cr.weaviate.io/semitechnologies/weaviate:latest
  ```
- [ ] Persistent volume configured for data storage
- [ ] Backup strategy implemented (daily exports)
- [ ] Memory allocation: >4GB RAM (for 100k documents)
- [ ] Network security: Firewall rules, API key authentication
- [ ] Connection pooling enabled
- [ ] Monitoring agents installed

### Document Storage
- [ ] Source documents organized in `data/` directory
- [ ] Backup system in place
- [ ] Document versioning tracked
- [ ] Access logs enabled

### Computing Resources
- [ ] CPU: 2+ cores for async processing
- [ ] RAM: 8+ GB (Weaviate + Python runtime)
- [ ] Disk: SSD recommended (vector I/O intensive)
- [ ] Network: Stable, <100ms latency to Weaviate

## Application Configuration

### Ingestion Pipeline
- [ ] Document chunking strategy selected (fixed-size vs semantic)
- [ ] Chunk size configured (512 tokens)
- [ ] Overlap configured (100 tokens)
- [ ] Metadata extraction working
- [ ] Embedding model selected (`text-embedding-3-small`)
- [ ] Batch ingestion working (for 1k+ documents)

### Retrieval Pipeline
- [ ] Vector search working (top-100 retrieval)
- [ ] Reranking enabled with cross-encoder
- [ ] Reranker model loaded (`ms-marco-MiniLM-L-12-v2`)
- [ ] Batch reranking implemented
- [ ] Fallback to vector-only if reranker fails

### Caching Layer
- [ ] Query cache initialized (max_size=1000)
- [ ] Similarity threshold tuned (0.95)
- [ ] Cache eviction policy working (LRU)
- [ ] Cache hit rate monitoring enabled
- [ ] TTL (time-to-live) implemented if needed

### Generation Pipeline
- [ ] LLM model selected (gpt-3.5-turbo)
- [ ] Temperature parameter tuned (0.7)
- [ ] Prompt template finalized
- [ ] Max tokens limited (to control costs)

## Testing & Validation

### Unit Tests
- [ ] Run `pytest week-2/test_*.py -v`
- [ ] All tests passing
- [ ] Coverage >80%

### Integration Tests
- [ ] Ingestion: document → chunks → embeddings → Weaviate
- [ ] Retrieval: query → vector search → reranking → results
- [ ] Caching: cache hit/miss scenarios
- [ ] End-to-end: ingestion + query + generation

### Quality Tests
- [ ] Evaluation metrics calculated
- [ ] MRR > 0.7 on test queries
- [ ] NDCG > 0.75 on test queries
- [ ] Reranking improves MRR by >10%

### Performance Tests
- [ ] Vector search: <100ms
- [ ] Reranking: <100ms per 100 results
- [ ] Cache hits: <5ms
- [ ] E2E latency: <2s (or <10ms with cache)

### Stress Tests
- [ ] 1000 concurrent queries
- [ ] Weaviate handles 100k+ documents
- [ ] Memory usage stable (no memory leaks)
- [ ] CPU usage reasonable (<80%)

## Monitoring & Observability

### Metrics Collection
- [ ] Cache hit rate tracked
- [ ] Retrieval MRR/NDCG tracked
- [ ] Latency per stage (embedding, search, rerank, llm)
- [ ] Document count in Weaviate
- [ ] API costs tracked (embeddings + LLM)

### Logging
- [ ] Debug logs configured
- [ ] Error logs to file/service
- [ ] Structured logging (JSON format)
- [ ] Log rotation implemented
- [ ] Log aggregation tool (ELK, Datadog, etc.)

### Dashboards
- [ ] Cache statistics dashboard
- [ ] Retrieval quality metrics
- [ ] Latency percentiles (p50, p95, p99)
- [ ] System resource usage (CPU, RAM, disk)
- [ ] API cost tracking

### Alerting
- [ ] Alert: Weaviate down
- [ ] Alert: Cache hit rate <20%
- [ ] Alert: Latency >1000ms
- [ ] Alert: MRR drops >5%
- [ ] Alert: Disk usage >80%

## Security

### Data Protection
- [ ] API keys stored in environment variables (not hardcoded)
- [ ] Weaviate API key/authentication enabled
- [ ] OpenAI API key secure
- [ ] Database backups encrypted
- [ ] HTTPS enabled (if exposed publicly)

### Access Control
- [ ] Authentication required for API
- [ ] Rate limiting implemented (e.g., 100 req/min per IP)
- [ ] Allowed origins configured (CORS)
- [ ] Admin endpoints protected

### Compliance
- [ ] Document retention policy defined
- [ ] GDPR compliance (if applicable)
  - [ ] Can delete user documents
  - [ ] Can export user data
- [ ] Data residency requirements met

## Documentation

### System Documentation
- [ ] Architecture diagram
- [ ] Data flow diagram
- [ ] Component interactions
- [ ] Deployment diagram

### Operational Documentation
- [ ] How to add documents
- [ ] How to update embeddings
- [ ] How to rebuild index
- [ ] How to recover from backup
- [ ] Troubleshooting guide

### Developer Documentation
- [ ] README with quick start
- [ ] API specification (endpoints, request/response)
- [ ] Configuration reference
- [ ] Performance tuning guide

## Cost Optimization

### API Spend
- [ ] Embedding batching configured (1k vectors at a time)
- [ ] Caching reducing repeated embeddings
- [ ] LLM model cost-effective (3.5-turbo vs 4)
- [ ] Streaming enabled (saves tokens on long outputs)
- [ ] Estimated monthly cost <$100

### Infrastructure Cost
- [ ] Weaviate sizing appropriate
- [ ] Disk usage <500MB (for test set)
- [ ] No unnecessary cloud services
- [ ] Auto-scaling configured (if cloud)

## Pre-Launch Verification

- [ ] All infrastructure ready
- [ ] All tests passing
- [ ] All metrics enabled
- [ ] Documentation complete
- [ ] Team trained on operations
- [ ] Runbook for common issues prepared
- [ ] Backup & recovery tested
- [ ] Load testing done (1000+ qpm)
- [ ] Security audit complete
- [ ] Cost estimate approved

## Launch

```bash
# Final verification
pytest week-2/ -v
python -m week-2.evaluate_retrieval
curl http://localhost:8080/v1/.well-known/ready

# If all passing → Ready for production
```

## Post-Launch (First Week)

- [ ] Monitor dashboard daily
- [ ] Check alerts daily
- [ ] Review logs for errors
- [ ] Measure real user query patterns
- [ ] Adjust cache threshold if needed
- [ ] Collect user feedback
- [ ] Document any issues

## Rollback Plan

If production issues occur:

1. **Degradation**: Turn off reranking (faster, less accurate)
2. **Further degradation**: Use vector search only
3. **Full rollback**: Switch to Week 1 in-memory system
4. **Investigation**: Debug in staging environment

Estimated time: 5-10 minutes to full rollback

---

## Key Metrics to Monitor

### Retrieval Quality
- MRR (target: >0.7)
- NDCG (target: >0.75)
- P@10 (target: >0.6)

### Performance
- Query latency p50 (target: <200ms)
- Query latency p99 (target: <1000ms)
- Cache hit rate (target: >30%)

### System Health
- Weaviate uptime (target: 99.9%)
- Error rate (target: <0.1%)
- API cost per query (target: <$0.01)

---

**Checklist Version**: 1.0  
**Last Updated**: January 5, 2026  
**Status**: Ready for production deployment
