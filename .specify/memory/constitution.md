<!-- Sync Impact Report -->
<!-- Version change: None (Initial Creation) -->
<!-- Modified principles: None -->
<!-- Added sections: Core Principles, Key Standards and Constraints, Development and Quality Gates, Governance -->
<!-- Removed sections: None -->
<!-- Templates requiring updates:
✅ .specify/templates/plan-template.md
✅ .specify/templates/spec-template.md
✅ .specify/templates/tasks-template.md
✅ .claude/commands/sp.adr.md
✅ .claude/commands/sp.analyze.md
✅ .claude/commands/sp.checklist.md
✅ .claude/commands/sp.clarify.md
✅ .claude/commands/sp.constitution.md
✅ .claude/commands/sp.git.commit_pr.md
✅ .claude/commands/sp.implement.md
✅ .claude/commands/sp.phr.md
✅ .claude/commands/sp.plan.md
✅ .claude/commands/sp.specify.md
✅ .claude/commands/sp.tasks.md
-->
<!-- Follow-up TODOs: None -->
# AI-native technical textbook on Physical AI & Humanoid Robotics Constitution

## Core Principles

### I. Technical Accuracy
All technical content MUST be based on official robotics documentation, prioritizing primary sources for factual correctness.

### II. Clarity for Engineering Students
Content MUST be clear, concise, and accessible to engineering students with computer science and robotics backgrounds, avoiding unnecessary jargon.

### III. Reproducibility
All code examples and setup instructions MUST be fully testable and reproducible, with clear steps for verification.

### IV. Docusaurus Consistency
Documentation MUST adhere to Docusaurus standards and best practices, following Context7 documentation guidelines for consistent formatting and structure.

## Key Standards and Constraints

All technical content MUST map to primary robotics sources.
Reference Priority:
1. ROS2 documentation
2. Gazebo documentation
3. NVIDIA Isaac documentation
4. Docusaurus documentation via Context7

Allowed Code Types:
- Python (ROS2 rclpy)
- URDF
- .launch XML files

Frontend: Markdown documentation under `/docs/`

Project Constraints:
- Total: 15 chapters
- Code Examples: 3+ per chapter
- Lab Exercises: One per chapter
- Rendering: Book MUST be rendered as a Docusaurus docs site following Context7 guidelines.

## Development and Quality Gates

Success Criteria:
- All chapters MUST render correctly with sidebar navigation.
- The Docusaurus build process MUST compile without errors.

## Governance

- Constitution supersedes all other project practices and documentation.
- Amendments to this constitution REQUIRE documentation, approval by project maintainers, and a clear migration plan for affected components.
- All code contributions and reviews MUST verify compliance with the principles and standards outlined herein.

**Version**: 1.0.0 | **Ratified**: 2025-12-08 | **Last Amended**: 2025-12-08
